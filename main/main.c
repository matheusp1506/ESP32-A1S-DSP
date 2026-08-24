#include "driver/gpio.h"
#include "driver/i2c.h"
#include "driver/i2s_std.h"
#include "esp_log.h"
#include "esp_system.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Bibliotecas ESP-DSP para Filtros

#include "dsps_biquad.h"
#include "dsps_fir.h"

static const char *TAG = "AUDIO";

#define I2C_MASTER_NUM I2C_NUM_0
#define I2C_SDA 33
#define I2C_SCL 32
#define GPIO_PA_EN 21

// Pinos do ESP32-A1S (Audio-Kit)
#define I2S_MCK_IO                                                             \
  0 // GPIO 0 envia o MCLK (256xFs = 12.288 MHz) necessário para o ADC do ES8388
#define I2S_BCK_IO 27
#define I2S_WS_IO 25
#define I2S_DO_IO 26
#define I2S_DI_IO 35

#define ES8388_ADC_PGA_GAIN                                                    \
  0x00 // 0 dB gain (0xC0 was +36dB for Mic, causing severe clipping with PC
       // line signals)

// Onboard Buttons for ESP32-A1S Audio Kit v2.2
#define BUTTON_KEY1_GPIO GPIO_NUM_36 // Key 1: FIR Float 32
#define BUTTON_KEY2_GPIO GPIO_NUM_13 // Key 2: FIR Fixed Q15
#define BUTTON_KEY3_GPIO GPIO_NUM_19 // Key 3: IIR Float Biquad
#define BUTTON_KEY4_GPIO GPIO_NUM_23 // Key 4: IIR Q15 Cascade
#define BUTTON_KEY5_GPIO GPIO_NUM_18 // Key 5: Passthrough
#define BUTTON_KEY6_GPIO GPIO_NUM_5  // Key 6: Pass Nothing (Mute)

static volatile int modo_filtro = 0;

static i2s_chan_handle_t tx_handle = NULL;
static i2s_chan_handle_t rx_handle = NULL;

#include "filter_coefficients.h"

static inline int16_t clamp_s16(int32_t val) {
  if (val > 32767)
    return 32767;
  if (val < -32768)
    return -32768;
  return (int16_t)val;
}

void process_iir_q15_cascade(const int16_t *src, int16_t *dst, int len,
                             iir_biquad_q15_stage_t *stages, int num_stages) {
  for (int i = 0; i < len; i++) {
    int32_t sample = src[i];

    for (int s = 0; s < num_stages; s++) {
      iir_biquad_q15_stage_t *st = &stages[s];

      // Difference Equation: y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1]
      // - a2*y[n-2]
      int64_t acc = 0;
      acc += (int64_t)st->b0 * sample;
      acc += (int64_t)st->b1 * st->x1;
      acc += (int64_t)st->b2 * st->x2;
      acc -= (int64_t)st->a1 * st->y1;
      acc -= (int64_t)st->a2 * st->y2;

      // Shift back by 14 bits for Q14 format
      int32_t out32 = (int32_t)(acc >> 14);

      // Saturate to 16-bit PCM output limits
      int16_t out16 = clamp_s16(out32);

      // Update state registers
      st->x2 = st->x1;
      st->x1 = (int16_t)sample;
      st->y2 = st->y1;
      st->y1 = out16;

      sample = out16;
    }

    dst[i] = (int16_t)sample;
  }
}

void process_fir_q15_direct(const int16_t *src, int16_t *dst, int len,
                            const int16_t *coeffs, int num_taps,
                            int16_t *delay) {
  // 1. Copia o novo bloco de entrada para a frente do histórico de atraso
  memcpy(&delay[num_taps - 1], src, len * sizeof(int16_t));

  // 2. Convolução direta (mesma ordem matemática do dsps_fir_f32)
  for (int i = 0; i < len; i++) {
    int64_t acc = 0;
    const int16_t *s = &delay[i];

    for (int k = 0; k < num_taps; k++) {
      acc += (int64_t)coeffs[k] * (int64_t)s[num_taps - 1 - k];
    }

    // Arredondamento simétrico + shift Q15
    int32_t out32 = (int32_t)((acc + 16384) >> 15);
    dst[i] = clamp_s16(out32);
  }

  // 3. Move o histórico de volta para o início do array
  memmove(delay, &delay[len], (num_taps - 1) * sizeof(int16_t));
}

static esp_err_t write_reg(uint8_t addr, uint8_t reg, uint8_t val) {
  uint8_t d[2] = {reg, val};
  esp_err_t err = i2c_master_write_to_device(I2C_MASTER_NUM, addr, d, 2,
                                             pdMS_TO_TICKS(100));

  if (err != ESP_OK) {
    ESP_LOGE(TAG, "Falha ao escrever reg 0x%02X: %s", reg,
             esp_err_to_name(err));
  }

  return err;
}

// ============================================================================
// INICIALIZAÇÃO DE REGISTRADORES PADRÃO ESPRESSIF (ES8388)
// ============================================================================

static void es8388_init_espressif_standard(uint8_t addr) {
  // 1. Reset
  write_reg(addr, 0x00, 0x80);
  vTaskDelay(pdMS_TO_TICKS(50));
  write_reg(addr, 0x00, 0x00);

  // 2. Slave mode
  write_reg(addr, 0x08, 0x00); // Slave mode (utiliza MCLK do pin GPIO 0)
  write_reg(addr, 0x02, 0xF3); // CHIPPOWER: power down DEM/STM durante config
  write_reg(addr, 0x2B, 0x80); // DACCONTROL21: mesmo LRCK para ADC/DAC
  write_reg(addr, 0x00, 0x14); // CONTROL1
  write_reg(addr, 0x01, 0x40); // CONTROL2: power up analog/Ibias

  // 3. ADC (Line-In)
  write_reg(addr, 0x03, 0x00); // Power Up ADC
  write_reg(addr, 0x0A, 0x00); // LIN1/RIN1 input
  write_reg(addr, 0x0B, 0x00); // Single-ended input mode (0x00) for standard
                               // Line-In (0xF0 was differential MIC mode)
  write_reg(
      addr, 0x09,
      ES8388_ADC_PGA_GAIN);    // 0dB input gain to prevent Line-In saturation
  write_reg(addr, 0x0C, 0x0C); // 16-bit I2S
  write_reg(addr, 0x0D, 0x02); // Ratio 256 (MCLK = 256 * Fs = 12.288 MHz)
  write_reg(addr, 0x10, 0x00); // Vol ADC L 0dB
  write_reg(addr, 0x11, 0x00); // Vol ADC R 0dB

  // 4. Power up total
  write_reg(addr, 0x01, 0x50); // CONTROL2
  write_reg(addr, 0x02, 0x00); // CHIPPOWER: power up total
  write_reg(addr, 0x08, 0x00); // Slave mode

  // 5. DAC
  write_reg(addr, 0x04, 0x3C); // Power Up DAC + LOUT1/ROUT1/LOUT2/ROUT2
  write_reg(addr, 0x00, 0x12); // CONTROL1
  write_reg(addr, 0x17, 0x18); // DACCONTROL1: 16-bit I2S
  write_reg(addr, 0x18, 0x02); // Ratio 256 (MCLK = 256 * Fs = 12.288 MHz)

  // 6. Mixer de saída - Desativa completamente qualquer bypass analógico direto
  // (LIN/RIN -> LOUT/ROUT)
  write_reg(addr, 0x26, 0x00); // DACCONTROL16: 0x00 desativa a rota analógica
                               // de LIN/RIN para os mixers
  write_reg(addr, 0x27, 0x90); // DACCONTROL17: habilita apenas LDAC -> LMIX
                               // (0dB), sem bypass analógico
  write_reg(addr, 0x28, 0x00); // DACCONTROL18: zera totalmente o ganho do
                               // bypass analógico esquerdo
  write_reg(addr, 0x29, 0x00); // DACCONTROL19: zera totalmente o ganho do
                               // bypass analógico direito
  write_reg(addr, 0x2A, 0x90); // DACCONTROL20: habilita apenas RDAC -> RMIX
                               // (0dB), sem bypass analógico
  write_reg(addr, 0x2B, 0x80); // DACCONTROL21: mesmo LRCK para ADC/DAC
  write_reg(addr, 0x2D, 0x00); // DACCONTROL23: vroi=0

  // 7. Volumes de saída
  write_reg(addr, 0x1A, 0x00); // DAC Vol Left 0dB
  write_reg(addr, 0x1B, 0x00); // DAC Vol Right 0dB
  write_reg(addr, 0x2E, 0x1E); // LOUT1 Vol
  write_reg(addr, 0x2F, 0x1E); // ROUT1 Vol
  write_reg(addr, 0x30, 0x1E); // LOUT2 Vol
  write_reg(addr, 0x31, 0x1E); // ROUT2 Vol

  // 8. Desmuta
  write_reg(addr, 0x19, 0x00); // Unmute DAC

  ESP_LOGI(TAG, "ES8388 Configurado!");
}

// ============================================================================
// DRIVER I2S MISTO (SLOT PHILIPS PADRÃO)
// ============================================================================

static void i2s_driver_init(void) {
  i2s_chan_config_t chan_cfg =
      I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
  i2s_new_channel(&chan_cfg, &tx_handle, &rx_handle);

  i2s_std_config_t std_cfg = {
      .clk_cfg =
          {
              .sample_rate_hz = SAMPLE_RATE,
              .clk_src = I2S_CLK_SRC_DEFAULT,
              .mclk_multiple = I2S_MCLK_MULTIPLE_256,
          },
      .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_16BIT,
                                                      I2S_SLOT_MODE_STEREO),
      .gpio_cfg =
          {
              .mclk = I2S_MCK_IO,
              .bclk = I2S_BCK_IO,
              .ws = I2S_WS_IO,
              .dout = I2S_DO_IO,
              .din = I2S_DI_IO,
          },
  };

  i2s_channel_init_std_mode(tx_handle, &std_cfg);
  i2s_channel_init_std_mode(rx_handle, &std_cfg);
  i2s_channel_enable(tx_handle);
  i2s_channel_enable(rx_handle);

  ESP_LOGI(TAG, "I2S Inicializado!");
}

// ============================================================================
// TASK DE PASSTHROUGH (SAMPLES BRUTOS)
// ============================================================================

void passthrough_task(void *pvParameters) {
  size_t bytes_read = 0;
  size_t bytes_written = 0;
  int16_t buffer[512];
  uint32_t log_counter = 0;

  while (1) {
    esp_err_t ret_read = i2s_channel_read(rx_handle, buffer, sizeof(buffer),
                                          &bytes_read, portMAX_DELAY);

    if (ret_read == ESP_OK && bytes_read > 0) {
      // Escreve direto na saída I2S
      i2s_channel_write(tx_handle, buffer, bytes_read, &bytes_written,
                        portMAX_DELAY);
      log_counter++;

      if (log_counter % 100 == 0) {
        int16_t max_val = 0;
        int sample_count = bytes_read / sizeof(int16_t);

        for (int i = 0; i < sample_count; i++) {
          int16_t abs_sample = abs(buffer[i]);
          if (abs_sample > max_val)
            max_val = abs_sample;
        }
        ESP_LOGI(TAG, "Sinal no Peak Level: %d / 32767", max_val);
      }
    }
  }
}

// ============================================================================
// DIAGNÓSTICO COMPARATIVO COMPLETO: FIR F32 vs FIR Q15
// ============================================================================

void run_fir_diagnostics(void) {
  ESP_LOGI(TAG, "==================================================");
  ESP_LOGI(TAG, "      INICIANDO DIAGNÓSTICO COMPARATIVO FIR       ");
  ESP_LOGI(TAG, "==================================================");

#define DIAG_LEN 256
#define DIAG_FS 48000.0f

  static int16_t in_q15[DIAG_LEN];
  static float in_f32[DIAG_LEN];

  static int16_t out_q15[DIAG_LEN];
  static float out_f32[DIAG_LEN];

  // --------------------------------------------------------------------------
  // TESTE 1: FREQUÊNCIA DE REJEIÇÃO (SENOIDE 1000 Hz - DEVE ZERAR/ATENUAR)
  // --------------------------------------------------------------------------
  ESP_LOGI(TAG, "--- TESTE 1: SENOIDE 1000 Hz (FAIXA DE CORTE / NOTCH) ---");
  for (int i = 0; i < DIAG_LEN; i++) {
    float val = sinf(2.0f * M_PI * 1000.0f * (float)i / DIAG_FS);
    in_f32[i] = val;
    in_q15[i] = (int16_t)(val * 16384.0f); // Amplitude ~16384 (-6 dBFS)
  }

  // Reinicializa filtros
  memset(fir_delay_f32, 0, sizeof(fir_delay_f32));
  memset(fir_delay_q15, 0, sizeof(fir_delay_q15));

  dsps_fir_init_f32(&fir_inst_f32, fir_coeffs_f32, fir_delay_f32,
                    FIR_F32_TAP_COUNT);
  // Array com os coeficientes espelhados temporalmente
  static int16_t fir_coeffs_q15_reversed[FIR_Q15_TAP_COUNT];

  for (int i = 0; i < FIR_Q15_TAP_COUNT; i++) {
    fir_coeffs_q15_reversed[i] = fir_coeffs_q15[FIR_Q15_TAP_COUNT - 1 - i];
  }

  // Inicializa o ESP-DSP passando o array REVERSO
  dsps_fird_init_s16(&fir_inst_q15, fir_coeffs_q15_reversed, fir_delay_q15,
                     FIR_Q15_TAP_COUNT, 1, 0, 0);

  // Executa múltiplos blocos para passar o transiente dos 511 taps
  for (int b = 0; b < 4; b++) {
    dsps_fir_f32(&fir_inst_f32, in_f32, out_f32, DIAG_LEN);
    process_fir_q15_direct(in_q15, out_q15, DIAG_LEN, fir_coeffs_q15,
                           FIR_Q15_TAP_COUNT, fir_delay_q15);
  }

  ESP_LOGI(TAG, "Amostras Estabilizadas (1 kHz):");
  for (int i = DIAG_LEN - 8; i < DIAG_LEN; i++) {
    int16_t f32_conv = (int16_t)(out_f32[i] * 16384.0f);
    ESP_LOGI(TAG, "[%3d] In: %6d | F32 Out: %6d | Q15 Out: %6d | Diff: %d", i,
             in_q15[i], f32_conv, out_q15[i], abs(f32_conv - out_q15[i]));
  }

  // --------------------------------------------------------------------------
  // TESTE 2: BANDA PASSANTE (SENOIDE 5000 Hz - DEVE PASSAR DIRETO)
  // --------------------------------------------------------------------------
  ESP_LOGI(TAG, "\n--- TESTE 2: SENOIDE 5000 Hz (BANDA PASSANTE 0 dB) ---");
  for (int i = 0; i < DIAG_LEN; i++) {
    float val = sinf(2.0f * M_PI * 5000.0f * (float)i / DIAG_FS);
    in_f32[i] = val;
    in_q15[i] = (int16_t)(val * 16384.0f);
  }

  memset(fir_delay_f32, 0, sizeof(fir_delay_f32));
  memset(fir_delay_q15, 0, sizeof(fir_delay_q15));

  dsps_fir_init_f32(&fir_inst_f32, fir_coeffs_f32, fir_delay_f32,
                    FIR_F32_TAP_COUNT);
  dsps_fird_init_s16(&fir_inst_q15, fir_coeffs_q15, fir_delay_q15,
                     FIR_Q15_TAP_COUNT, 1, 0, 0);

  for (int b = 0; b < 4; b++) {
    dsps_fir_f32(&fir_inst_f32, in_f32, out_f32, DIAG_LEN);
    process_fir_q15_direct(in_q15, out_q15, DIAG_LEN, fir_coeffs_q15,
                           FIR_Q15_TAP_COUNT, fir_delay_q15);
  }

  ESP_LOGI(TAG, "Amostras Estabilizadas (5 kHz):");
  for (int i = DIAG_LEN - 8; i < DIAG_LEN; i++) {
    int16_t f32_conv = (int16_t)(out_f32[i] * 16384.0f);
    ESP_LOGI(TAG, "[%3d] In: %6d | F32 Out: %6d | Q15 Out: %6d | Diff: %d", i,
             in_q15[i], f32_conv, out_q15[i], abs(f32_conv - out_q15[i]));
  }

  // --------------------------------------------------------------------------
  // TESTE 3: DEGRAU DC (AMPLITUDE CONSTANTE 10000)
  // --------------------------------------------------------------------------
  ESP_LOGI(TAG, "\n--- TESTE 3: DEGRAU DC (Esperado: ~10000) ---");
  for (int i = 0; i < DIAG_LEN; i++) {
    in_f32[i] = 10000.0f / 32768.0f;
    in_q15[i] = 10000;
  }

  memset(fir_delay_f32, 0, sizeof(fir_delay_f32));
  memset(fir_delay_q15, 0, sizeof(fir_delay_q15));

  dsps_fir_init_f32(&fir_inst_f32, fir_coeffs_f32, fir_delay_f32,
                    FIR_F32_TAP_COUNT);
  dsps_fird_init_s16(&fir_inst_q15, fir_coeffs_q15, fir_delay_q15,
                     FIR_Q15_TAP_COUNT, 1, 0, 0);

  for (int b = 0; b < 4; b++) {
    dsps_fir_f32(&fir_inst_f32, in_f32, out_f32, DIAG_LEN);
    process_fir_q15_direct(in_q15, out_q15, DIAG_LEN, fir_coeffs_q15,
                           FIR_Q15_TAP_COUNT, fir_delay_q15);
  }

  int16_t dc_f32_conv = (int16_t)(out_f32[DIAG_LEN - 1] * 32768.0f);
  ESP_LOGI(TAG, "DC In: 10000 | F32 Out: %d | Q15 Out: %d | Diff: %d",
           dc_f32_conv, out_q15[DIAG_LEN - 1],
           abs(dc_f32_conv - out_q15[DIAG_LEN - 1]));

  // Limpeza dos buffers de delay para operação normal
  memset(fir_delay_f32, 0, sizeof(fir_delay_f32));
  memset(fir_delay_q15, 0, sizeof(fir_delay_q15));
  ESP_LOGI(TAG, "==================================================\n");
}

// ============================================================================
// TASK DE CONTROLE DOS BOTOES
// ============================================================================

void button_task(void *pvParameters) {
  // KEY1 is GPIO36 (Input-only, external pull-up)
  // KEY2..KEY6 (GPIOs 13, 19, 23, 18, 5) configured with pull-ups
  gpio_config_t btn_conf = {
      .pin_bit_mask = (1ULL << BUTTON_KEY1_GPIO) | (1ULL << BUTTON_KEY2_GPIO) |
                      (1ULL << BUTTON_KEY3_GPIO) | (1ULL << BUTTON_KEY4_GPIO) |
                      (1ULL << BUTTON_KEY5_GPIO) | (1ULL << BUTTON_KEY6_GPIO),
      .mode = GPIO_MODE_INPUT,
      .pull_up_en = GPIO_PULLUP_ENABLE,
      .pull_down_en = GPIO_PULLDOWN_DISABLE,
      .intr_type = GPIO_INTR_DISABLE,
  };
  gpio_config(&btn_conf);

  int last_key1_state = 1;
  int last_key2_state = 1;
  int last_key3_state = 1;
  int last_key4_state = 1;
  int last_key5_state = 1;
  int last_key6_state = 1;

  const char *modo_names[] = {"0: FIR Float 32",     "1: FIR Fixed Q15",
                              "2: IIR Float Biquad", "3: IIR Q15 Cascade",
                              "4: Passthrough",      "5: Pass Nothing (Mute)"};

  ESP_LOGI(TAG, "Task de Botoes iniciada.");
  ESP_LOGI(TAG, " KEY1 (GPIO36) -> FIR Float 32");
  ESP_LOGI(TAG, " KEY2 (GPIO13) -> FIR Fixed Q15");
  ESP_LOGI(TAG, " KEY3 (GPIO19) -> IIR Float Biquad");
  ESP_LOGI(TAG, " KEY4 (GPIO23) -> IIR Q15 Cascade");
  ESP_LOGI(TAG, " KEY5 (GPIO18) -> Passthrough");
  ESP_LOGI(TAG, " KEY6 (GPIO5)  -> Pass Nothing (Mute)");

  while (1) {
    int key1_state = gpio_get_level(BUTTON_KEY1_GPIO);
    int key2_state = gpio_get_level(BUTTON_KEY2_GPIO);
    int key3_state = gpio_get_level(BUTTON_KEY3_GPIO);
    int key4_state = gpio_get_level(BUTTON_KEY4_GPIO);
    int key5_state = gpio_get_level(BUTTON_KEY5_GPIO);
    int key6_state = gpio_get_level(BUTTON_KEY6_GPIO);

    // KEY1 -> Mode 0: FIR Float 32
    if (last_key1_state == 1 && key1_state == 0) {
      vTaskDelay(pdMS_TO_TICKS(50));
      if (gpio_get_level(BUTTON_KEY1_GPIO) == 0) {
        modo_filtro = 0;
        ESP_LOGI(TAG, ">>> Modo do Filtro alterado para: %s <<<",
                 modo_names[0]);
      }
    }

    // KEY2 -> Mode 1: FIR Fixed Q15
    if (last_key2_state == 1 && key2_state == 0) {
      vTaskDelay(pdMS_TO_TICKS(50));
      if (gpio_get_level(BUTTON_KEY2_GPIO) == 0) {
        modo_filtro = 1;
        ESP_LOGI(TAG, ">>> Modo do Filtro alterado para: %s <<<",
                 modo_names[1]);
      }
    }

    // KEY3 -> Mode 2: IIR Float Biquad
    if (last_key3_state == 1 && key3_state == 0) {
      vTaskDelay(pdMS_TO_TICKS(50));
      if (gpio_get_level(BUTTON_KEY3_GPIO) == 0) {
        modo_filtro = 2;
        ESP_LOGI(TAG, ">>> Modo do Filtro alterado para: %s <<<",
                 modo_names[2]);
      }
    }

    // KEY4 -> Mode 3: IIR Q15 Cascade
    if (last_key4_state == 1 && key4_state == 0) {
      vTaskDelay(pdMS_TO_TICKS(50));
      if (gpio_get_level(BUTTON_KEY4_GPIO) == 0) {
        modo_filtro = 3;
        ESP_LOGI(TAG, ">>> Modo do Filtro alterado para: %s <<<",
                 modo_names[3]);
      }
    }

    // KEY5 -> Mode 4: Passthrough
    if (last_key5_state == 1 && key5_state == 0) {
      vTaskDelay(pdMS_TO_TICKS(50));
      if (gpio_get_level(BUTTON_KEY5_GPIO) == 0) {
        modo_filtro = 4;
        ESP_LOGI(TAG, ">>> Modo do Filtro alterado para: %s <<<",
                 modo_names[4]);
      }
    }

    // KEY6 -> Mode 5: Pass Nothing (Mute)
    if (last_key6_state == 1 && key6_state == 0) {
      vTaskDelay(pdMS_TO_TICKS(50));
      if (gpio_get_level(BUTTON_KEY6_GPIO) == 0) {
        modo_filtro = 5;
        ESP_LOGI(TAG, ">>> Modo do Filtro alterado para: %s <<<",
                 modo_names[5]);
      }
    }

    last_key1_state = key1_state;
    last_key2_state = key2_state;
    last_key3_state = key3_state;
    last_key4_state = key4_state;
    last_key5_state = key5_state;
    last_key6_state = key6_state;

    vTaskDelay(pdMS_TO_TICKS(20)); // Polling rate
  }
}

#define INPUT_HEADROOM 1.0f

void audio_dsp_task(void *pvParameters) {
  size_t bytes_read = 0;
  size_t bytes_written = 0;

  // static: fora da pilha da task, evita estouro de stack
  static int16_t i2s_rx_raw[SAMPLES_PER_BUFFER * 2];
  static int16_t i2s_tx_raw[SAMPLES_PER_BUFFER * 2];

  static __attribute__((aligned(16))) float float_in[SAMPLES_PER_BUFFER];
  static __attribute__((aligned(16))) float float_out[SAMPLES_PER_BUFFER];
  static __attribute__((aligned(16))) int16_t q15_in[SAMPLES_PER_BUFFER];
  static __attribute__((aligned(16))) int16_t q15_out[SAMPLES_PER_BUFFER];

  // Zera os buffers de atraso
  memset(fir_delay_f32, 0, sizeof(fir_delay_f32));
  memset(fir_delay_q15, 0, sizeof(fir_delay_q15));
  memset(iir_delay_f32, 0, sizeof(iir_delay_f32));

  // Initialize Q15 IIR cascade
  for (int s = 0; s < IIR_STAGES; s++) {
    iir_stages_q15[s].b0 = iir_coeffs_q15_raw[s][0];
    iir_stages_q15[s].b1 = iir_coeffs_q15_raw[s][1];
    iir_stages_q15[s].b2 = iir_coeffs_q15_raw[s][2];
    iir_stages_q15[s].a1 = iir_coeffs_q15_raw[s][3];
    iir_stages_q15[s].a2 = iir_coeffs_q15_raw[s][4];
    iir_stages_q15[s].x1 = 0;
    iir_stages_q15[s].x2 = 0;
    iir_stages_q15[s].y1 = 0;
    iir_stages_q15[s].y2 = 0;
  }
  ESP_LOGI(TAG, "IIR Q15 Custom Cascade initialized!");

  for (int i = 0; i < FIR_F32_TAP_COUNT; i++) {
    fir_coeffs_f32[i] = (float)fir_coeffs_q15[i] / 32768.0f;
  }

  // --- FIR Float32 ---
  esp_err_t ret = dsps_fir_init_f32(&fir_inst_f32, fir_coeffs_f32,
                                    fir_delay_f32, FIR_F32_TAP_COUNT);

  if (ret != ESP_OK) {
    ESP_LOGE(TAG, "FIR F32 init failed: 0x%x", ret);
  } else {
    ESP_LOGI(TAG, "FIR F32 init successful (%d taps)", FIR_F32_TAP_COUNT);
  }

  // --- FIR Fixed-Point Q15 ---
  esp_err_t ret_q15 =
      dsps_fird_init_s16(&fir_inst_q15, fir_coeffs_q15, fir_delay_q15,
                         FIR_Q15_TAP_COUNT, 1, -1, 0);

  // Coeffs sum (diagnostics):
  float sum = 0;
  for (int i = 0; i < FIR_Q15_TAP_COUNT; i++) {
    sum += fir_coeffs_q15[i];
  }
  ESP_LOGI(TAG, "FIR Q15 coeffs sum: %f (taps: %d)", sum, FIR_Q15_TAP_COUNT);

  sum = 0;
  for (int i = 0; i < FIR_F32_TAP_COUNT; i++) {
    sum += fir_coeffs_f32[i];
  }
  ESP_LOGI(TAG, "FIR F32 coeffs sum: %f (taps: %d)", sum, FIR_F32_TAP_COUNT);

  if (ret_q15 != ESP_OK) {
    ESP_LOGE(TAG, "FIR Q15 init failed: 0x%x", ret_q15);
  } else {
    ESP_LOGI(TAG, "FIR Q15 init successful");
  }

  ESP_LOGI(TAG, "Iniciando Loop de Processamento PDS...");
  ESP_LOGI(TAG, "RX Buffer Addr: %p, TX Buffer Addr: %p", i2s_rx_raw,
           i2s_tx_raw);

  float *src = float_in;
  float *dst = float_out;

  while (1) {
    esp_err_t err_r = i2s_channel_read(
        rx_handle, i2s_rx_raw, sizeof(i2s_rx_raw), &bytes_read, portMAX_DELAY);

    if (err_r != ESP_OK || bytes_read == 0) {
      continue; // evita processar lixo se a leitura falhar
    }

    int samples_count = bytes_read / (2 * sizeof(int16_t));

    int16_t peak_left = 0;
    int16_t peak_right = 0;

    for (int i = 0; i < samples_count; i++) {
      int16_t left_sample = i2s_rx_raw[i * 2];
      int16_t right_sample = i2s_rx_raw[i * 2 + 1];

      if (abs(left_sample) > peak_left) {
        peak_left = abs(left_sample);
      }

      if (abs(right_sample) > peak_right) {
        peak_right = abs(right_sample);
      }

      q15_in[i] = (int16_t)(((int32_t)left_sample + (int32_t)right_sample) / 2);
      float_in[i] = ((float)q15_in[i] / 32768.0f) * INPUT_HEADROOM;
    }

    switch (modo_filtro) {
    case 0: // FIR Float
      dsps_fir_f32(&fir_inst_f32, float_in, float_out, samples_count);
      break;
    case 1: // FIR Ponto Fixo Q15
      /*{
        static int fir_q15_pos = 0;
        process_fir_q15(q15_in, q15_out, samples_count, fir_coeffs_q15,
                        FIR_TAP_COUNT, fir_delay_q15, &fir_q15_pos);

        static int contador_fir_q15 = 0;
        if (++contador_fir_q15 % 100 == 0) {
          int16_t max_in = 0, max_out = 0;
          for (int i = 0; i < samples_count; i++) {
            if (abs(q15_in[i]) > max_in)
              max_in = abs(q15_in[i]);
            if (abs(q15_out[i]) > max_out)
              max_out = abs(q15_out[i]);
          }
          ESP_LOGI(TAG, "FIR Q15 Peak IN: %d, Peak OUT: %d", max_in, max_out);
        }
      }*/
      /*dsps_fird_s16(&fir_inst_q15, q15_in, q15_out, samples_count);

      static int dbg_count = 0;
      if (++dbg_count % 100 == 0) {
        int16_t max_in = 0, max_out = 0;
        for (int i = 0; i < samples_count; i++) {
          if (abs(q15_in[i]) > max_in)
            max_in = abs(q15_in[i]);
          if (abs(q15_out[i]) > max_out)
            max_out = abs(q15_out[i]);
        }
        ESP_LOGI(TAG, "Q15 Streaming -> Peak IN: %d | Peak OUT: %d", max_in,
                 max_out);
      }*/
      process_fir_q15_direct(q15_in, q15_out, samples_count, fir_coeffs_q15,
                             FIR_Q15_TAP_COUNT, fir_delay_q15);
      break;

    case 2: // IIR Float (Biquad)

      // 1. Copy initial input to float_out
      memcpy(float_out, float_in, samples_count * sizeof(float));

      // 2. Cascade stages sequentially using ping-pong intermediate buffers
      for (int stage = 0; stage < IIR_STAGES; stage++) {
        float *coeffs = &iir_coeffs_f32[stage * 5];
        float *delay = &iir_delay_f32[stage * 2];

        // Process through the current stage
        dsps_biquad_f32(src, dst, samples_count, coeffs, delay);

        // Swap pointers for next stage (if more than 1 stage)
        src = dst;

        dst = (dst == float_out) ? float_in : float_out;
      }

      if (dst == float_in) {
        // If the last stage wrote to float_in, copy it back to float_out
        memcpy(float_out, float_in, samples_count * sizeof(float));
      }

      // 3. Safety Check: ONLY reset on true numerical errors (NaN or Infinity)

      for (int i = 0; i < samples_count; i++) {
        if (isnan(float_out[i]) || isinf(float_out[i])) {
          memset(iir_delay_f32, 0, sizeof(iir_delay_f32));
          float_out[i] = 0.0f;
          ESP_LOGW(TAG, "IIR NaN/Inf detected! Delay buffer reset.");
          break;
        }
      }

      static int contador_iir = 0;

      if (++contador_iir % 100 == 0) {
        float max_iir = 0;
        for (int i = 0; i < samples_count; i++) {
          float abs_val = fabsf(float_out[i]);
          if (abs_val > max_iir)
            max_iir = abs_val;
        }
        ESP_LOGI(TAG, "Pico de saida IIR (float_out): %f", max_iir);
      }
      break;

    case 3: // IIR Q15 Cascade
      process_iir_q15_cascade(q15_in, q15_out, samples_count, iir_stages_q15,
                              IIR_STAGES);

      static int contador_iir_q15 = 0;
      if (++contador_iir_q15 % 100 == 0) {
        int16_t max_q15 = 0;
        for (int i = 0; i < samples_count; i++) {
          int16_t abs_val = abs(q15_out[i]);
          if (abs_val > max_q15)
            max_q15 = abs_val;
        }
        ESP_LOGI(TAG, "Pico de saída IIR Q15 Custom: %d / 32767", max_q15);
      }
      break;
    case 4: // Passthrough (copia entrada direto para a saída em q15)
      memcpy(q15_out, q15_in, samples_count * sizeof(int16_t));
      break;
    case 5: // Pass Nothing (Mute: zera a saída em ambos os buffers)
      memset(q15_out, 0, samples_count * sizeof(int16_t));
      memset(float_out, 0, samples_count * sizeof(float));
      break;
    default:
      break;
    }

    for (int i = 0; i < samples_count; i++) {
      int16_t sample_out = 0;

      if (modo_filtro == 0 || modo_filtro == 2) {
        float val = float_out[i] * 32768.0f;

        if (val > 32767.0f)
          val = 32767.0f;

        if (val < -32768.0f)
          val = -32768.0f;

        sample_out = (int16_t)val;
      } else {
        sample_out = q15_out[i];
      }

      i2s_tx_raw[i * 2] = sample_out;
      i2s_tx_raw[i * 2 + 1] = sample_out;
    }

    i2s_channel_write(tx_handle, i2s_tx_raw, bytes_read, &bytes_written,
                      portMAX_DELAY);
  }
}

void app_main(void) {
  // 1. Liga o Amplificador (PA)
  gpio_config_t io_conf = {
      .pin_bit_mask = (1ULL << GPIO_PA_EN),
      .mode = GPIO_MODE_OUTPUT,
  };

  gpio_config(&io_conf);
  gpio_set_level(GPIO_PA_EN, 1);

  // 2. Inicializa I2C
  i2c_config_t conf = {.mode = I2C_MODE_MASTER,
                       .sda_io_num = I2C_SDA,
                       .scl_io_num = I2C_SCL,
                       .sda_pullup_en = GPIO_PULLUP_ENABLE,
                       .scl_pullup_en = GPIO_PULLUP_ENABLE,
                       .master.clk_speed = 100000};

  i2c_param_config(I2C_MASTER_NUM, &conf);
  i2c_driver_install(I2C_MASTER_NUM, conf.mode, 0, 0, 0);

  // 3. Inicializa I2S (Inicia MCLK no GPIO 0)
  i2s_driver_init();
  vTaskDelay(pdMS_TO_TICKS(100));

  // 4. Configura ES8388
  es8388_init_espressif_standard(0x10);

  run_fir_diagnostics();

  // 5. Inicia Task
  // xTaskCreatePinnedToCore(passthrough_task, "passthrough_task", 4096, NULL,
  // 5, NULL, 1);
  xTaskCreatePinnedToCore(button_task, "button_task", 2048, NULL, 3, NULL, 0);
  xTaskCreatePinnedToCore(audio_dsp_task, "audio_dsp_task", 8192, NULL, 5, NULL,
                          1);
}