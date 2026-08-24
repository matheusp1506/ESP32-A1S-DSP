#!/usr/bin/env python3
"""
Gerador de Coeficientes de Filtros FIR e IIR para ESP32-DSP
==========================================================
Este script projeta filtros FIR e IIR (Ponto Flutuante 32-bit e Ponto Fixo Q15/Q14)
e gera o código C formatado exatamente como esperado pelo arquivo `main/filter_coefficients.h`.

Requisitos:
    numpy, scipy

Uso:
    python3 external/main.py
    python3 external/main.py --fir-f32-taps 511 --fir-q15-taps 255 --iir-stages 4 --cutoff 600 1100 --type bandstop
    python3 external/main.py --update-main

Atalho para definir ambos os FIR com o mesmo número de taps:
    python3 external/main.py --fir-taps 511  (equivale a --fir-f32-taps 511 --fir-q15-taps 511)
"""

import argparse
import re
import sys
import numpy as np
import scipy.signal as signal

# ==============================================================================
# UTILITÁRIOS
# ==============================================================================

def clamp_int16(val):
    """Satura o valor inteiro para a faixa de 16 bits com sinal [-32768, 32767]."""
    return int(np.clip(val, -32768, 32767))


# ==============================================================================
# SEÇÃO A: FIR PONTO FLUTUANTE (Float32)
# ==============================================================================

def design_fir_f32(numtaps=511, cutoff=[600, 1100], fs=48000,
                   pass_zero='bandstop', window='hamming'):
    """
    Projeta um filtro FIR Float32 usando o método da janela (firwin).

    Parâmetros:
        numtaps   : número de coeficientes (taps) do filtro
        cutoff    : frequência(s) de corte em Hz
        fs        : frequência de amostragem em Hz
        pass_zero : tipo do filtro ('lowpass', 'highpass', 'bandpass', 'bandstop')
        window    : janela de apodização ('hamming', 'hann', 'blackman', etc.)

    Retorna:
        coeffs_f32 (np.ndarray, float32): coeficientes em ponto flutuante
    """
    coeffs = signal.firwin(numtaps, cutoff, pass_zero=pass_zero,
                           window=window, fs=fs)
    return coeffs.astype(np.float32)


# ==============================================================================
# SEÇÃO B: FIR PONTO FIXO (Q15 — int16)
# ==============================================================================

def design_fir_q15(numtaps=511, cutoff=[600, 1100], fs=48000,
                   pass_zero='bandstop', window='hamming'):
    """
    Projeta um filtro FIR em ponto fixo Q15 usando o método da janela (firwin).

    Parâmetros:
        numtaps   : número de coeficientes (taps) do filtro
        cutoff    : frequência(s) de corte em Hz
        fs        : frequência de amostragem em Hz
        pass_zero : tipo do filtro ('lowpass', 'highpass', 'bandpass', 'bandstop')
        window    : janela de apodização ('hamming', 'hann', 'blackman', etc.)

    Retorna:
        coeffs_q15 (np.ndarray, int16): coeficientes quantizados em Q15 (×32768)
    """
    coeffs = signal.firwin(numtaps, cutoff, pass_zero=pass_zero,
                           window=window, fs=fs)
    coeffs_f32 = coeffs.astype(np.float32)
    # Quantização Q15: multiplica por 32768 e satura para int16
    coeffs_q15 = np.array(
        [clamp_int16(np.round(c * 32768.0)) for c in coeffs_f32],
        dtype=np.int16
    )
    return coeffs_q15


# ==============================================================================
# SEÇÃO C/D: IIR PONTO FLUTUANTE E FIXO (Biquad / SOS)
# ==============================================================================

def design_iir(stages=4, cutoff=[600, 1100], fs=48000,
               btype='bandstop', ftype='butter'):
    """
    Projeta um filtro IIR usando Seções de Segunda Ordem (SOS / Biquad).

    Retorna:
        coeffs_f32     (np.ndarray, float32): [b0, b1, b2, a1, a2] por estágio
                       formatado para ESP-DSP dsps_biquad_f32
        coeffs_q15_raw (np.ndarray, int16):  coeficientes em Q14 (×16384) por estágio
                       no formato [[b0, b1, b2, a1, a2], ...]
        actual_stages  (int): número real de estágios SOS gerados
    """
    sos = signal.iirfilter(N=stages, Wn=cutoff, btype=btype, ftype=ftype,
                           fs=fs, output='sos')
    actual_stages = sos.shape[0]

    coeffs_f32_list  = []
    coeffs_q15_list  = []

    # Cada linha do sos no SciPy é: [b0, b1, b2, a0, a1, a2] com a0 = 1.0
    # Para o ESP-DSP dsps_biquad_f32, a ordem por estágio é [b0, b1, b2, a1, a2]
    for stage in sos:
        b0, b1, b2, a0, a1, a2 = stage

        # Normalização por a0
        b0 /= a0; b1 /= a0; b2 /= a0
        a1 /= a0; a2 /= a0

        coeffs_f32_list.extend([b0, b1, b2, a1, a2])

        # Quantização Q14 para IIR Biquad (×16384)
        # Suporta valores no intervalo [-2.0, +1.9999] em int16_t
        q14_row = [
            clamp_int16(np.round(b0 * 16384.0)),
            clamp_int16(np.round(b1 * 16384.0)),
            clamp_int16(np.round(b2 * 16384.0)),
            clamp_int16(np.round(a1 * 16384.0)),
            clamp_int16(np.round(a2 * 16384.0)),
        ]
        coeffs_q15_list.append(q14_row)

    return (np.array(coeffs_f32_list, dtype=np.float32),
            np.array(coeffs_q15_list, dtype=np.int16),
            actual_stages)


# ==============================================================================
# FORMATAÇÃO DO CÓDIGO C (filter_coefficients.h)
# ==============================================================================

def format_c_code(fir_f32, fir_q15, iir_f32, iir_q15, sample_rate=48000):
    """
    Formata coeficientes FIR (F32 e Q15, independentes) e IIR para o arquivo
    main/filter_coefficients.h, com cada seção claramente delimitada.
    """
    fir_f32_tap_count = len(fir_f32)
    fir_q15_tap_count = len(fir_q15)
    iir_stages        = len(iir_q15)

    L = []   # linhas do arquivo de saída

    L += [
        "#ifndef FILTER_COEFFICIENTS_H",
        "#define FILTER_COEFFICIENTS_H",
        "",
        '#include "dsps_fir.h"',
        "#include <stdint.h>",
        "",
        "// ============================================================================",
        "// PARÂMETROS GLOBAIS DE SISTEMA",
        "// ============================================================================",
        "",
        f"#define SAMPLE_RATE        {sample_rate}",
        "#define SAMPLES_PER_BUFFER 256",
    ]

    # ------------------------------------------------------------------
    # SEÇÃO A — FIR Float32
    # ------------------------------------------------------------------
    L += [
        "",
        "// ============================================================================",
        "// SEÇÃO A: FIR PONTO FLUTUANTE (32-bit Float)",
        "// ============================================================================",
        f"// Gerado por: python3 external/main.py --fir-f32-taps {fir_f32_tap_count} --type <tipo> --cutoff <fc>",
        "// Para regenerar apenas esta seção: design_fir_f32() em external/main.py",
        "",
        f"#define FIR_F32_TAP_COUNT {fir_f32_tap_count}",
        "",
        "static fir_f32_t fir_inst_f32;",
        "__attribute__((aligned(16))) static float fir_coeffs_f32[FIR_F32_TAP_COUNT] = {",
    ]
    f32_strs = [f"{v:.9f}f" for v in fir_f32]
    for i in range(0, len(f32_strs), 5):
        chunk = f32_strs[i:i+5]
        comma = "," if (i + 5) < len(f32_strs) else ""
        L.append("    " + ", ".join(chunk) + comma)
    L += [
        "};",
        "",
        "__attribute__((aligned(16))) static float fir_delay_f32[FIR_F32_TAP_COUNT + 4];",
    ]

    # ------------------------------------------------------------------
    # SEÇÃO B — FIR Q15
    # ------------------------------------------------------------------
    L += [
        "",
        "// ============================================================================",
        "// SEÇÃO B: FIR PONTO FIXO (16-bit Q15)",
        "// ============================================================================",
        f"// Gerado por: python3 external/main.py --fir-q15-taps {fir_q15_tap_count} --type <tipo> --cutoff <fc>",
        "// Para regenerar apenas esta seção: design_fir_q15() em external/main.py",
        "",
        f"#define FIR_Q15_TAP_COUNT {fir_q15_tap_count}",
        "",
        "static fir_s16_t fir_inst_q15;",
        "__attribute__((aligned(16))) static int16_t fir_coeffs_q15[FIR_Q15_TAP_COUNT] = {",
    ]
    q15_strs = [f"{v:d}" for v in fir_q15]
    for i in range(0, len(q15_strs), 12):
        chunk = q15_strs[i:i+12]
        comma = "," if (i + 12) < len(q15_strs) else ""
        L.append("    " + ", ".join(chunk) + comma)
    L += [
        "};",
        "",
        "__attribute__((aligned(16))) static int16_t",
        "    fir_delay_q15[FIR_Q15_TAP_COUNT + SAMPLES_PER_BUFFER + 16] = {0};",
    ]

    # ------------------------------------------------------------------
    # SEÇÃO C — IIR Float32
    # ------------------------------------------------------------------
    L += [
        "",
        "// ============================================================================",
        "// SEÇÃO C: IIR PONTO FLUTUANTE (Biquad Float32)",
        "// ============================================================================",
        f"// Gerado por: python3 external/main.py --iir-stages {iir_stages} --type <tipo> --cutoff <fc>",
        "// Para regenerar apenas esta seção: design_iir() em external/main.py",
        "",
        f"#define IIR_STAGES {iir_stages}",
        "",
        "// Estrutura [b0, b1, b2, a1, a2] pronta para o ESP-DSP dsps_biquad_f32",
        "static float iir_coeffs_f32[5 * IIR_STAGES] = {",
    ]
    iir_f32_strs = [f"{v:.9f}f" for v in iir_f32]
    for i in range(0, len(iir_f32_strs), 5):
        chunk = iir_f32_strs[i:i+5]
        comma = "," if (i + 5) < len(iir_f32_strs) else ""
        L.append("    " + ", ".join(chunk) + comma)
    L += [
        "};",
        "",
        "static float iir_delay_f32[2 * IIR_STAGES] = {0};",
    ]

    # ------------------------------------------------------------------
    # SEÇÃO D — IIR Q14 (Biquad Cascade Custom)
    # ------------------------------------------------------------------
    L += [
        "",
        "// ============================================================================",
        "// SEÇÃO D: IIR PONTO FIXO (Q14 — Biquad Cascade Custom)",
        "// ============================================================================",
        "// Quantizado a partir dos coeficientes float: integer = (int16_t)round(float * 16384.0f)",
        "// Para regenerar: design_iir() em external/main.py",
        "",
        "// Estrutura de um único estágio Biquad Q14",
        "typedef struct {",
        "    int16_t b0, b1, b2; // Coeficientes do numerador (Q14)",
        "    int16_t a1, a2;     // Coeficientes do denominador (Q14)",
        "    int16_t x1, x2;     // Memória de estado — entrada",
        "    int16_t y1, y2;     // Memória de estado — saída",
        "} iir_biquad_q15_stage_t;",
        "",
        "// Vetor de estágios (estado inicializado em audio_dsp_task)",
        "static iir_biquad_q15_stage_t iir_stages_q15[IIR_STAGES];",
        "",
        "// Coeficientes Q14 por estágio [b0, b1, b2, a1, a2]",
        "static const int16_t iir_coeffs_q15_raw[IIR_STAGES][5] = {",
        "    //  b0,     b1,     b2,     a1,     a2",
    ]
    for idx, stage in enumerate(iir_q15):
        comma = "," if idx < len(iir_q15) - 1 else ""
        row_str = ", ".join(f"{v:d}" for v in stage)
        L.append(f"    {{{row_str}}}{comma}")
    L += [
        "};",
        "",
        "#endif // FILTER_COEFFICIENTS_H",
    ]

    return "\n".join(L)


# ==============================================================================
# ATUALIZAÇÃO DO ARQUIVO DE COEFICIENTES
# ==============================================================================

def update_header_file(c_code, target_path="main/filter_coefficients.h"):
    """
    Atualiza o arquivo de coeficientes de filtro (main/filter_coefficients.h).
    """
    try:
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(c_code + "\n")
        print(f"Sucesso: '{target_path}' foi atualizado com os novos coeficientes!",
              file=sys.stderr)
    except Exception as e:
        print(f"Erro ao atualizar '{target_path}': {e}", file=sys.stderr)


# ==============================================================================
# PONTO DE ENTRADA
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Gera coeficientes de filtros FIR e IIR formatados para o ESP32",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--fs", type=int, default=48000,
                        help="Frequência de amostragem em Hz (padrão: 48000)")

    # --- FIR taps (independentes por tipo) ---
    parser.add_argument("--fir-taps", type=int, default=None,
                        help="Atalho: define --fir-f32-taps e --fir-q15-taps com o mesmo valor")
    parser.add_argument("--fir-f32-taps", type=int, default=511,
                        help="Número de taps do FIR Float32 (padrão: 511)")
    parser.add_argument("--fir-q15-taps", type=int, default=511,
                        help="Número de taps do FIR Q15 (padrão: 511)")

    parser.add_argument("--iir-stages", type=int, default=4,
                        help="Número de estágios Biquad do IIR (padrão: 4)")
    parser.add_argument("--cutoff", type=float, nargs="+", default=[600.0, 1100.0],
                        help="Frequências de corte em Hz (ex: --cutoff 600 1100)")
    parser.add_argument("--type", type=str, default="bandstop",
                        choices=["bandstop", "bandpass", "lowpass", "highpass"],
                        help="Tipo de filtro (padrão: bandstop)")
    parser.add_argument("--output", type=str, default=None,
                        help="Caminho do arquivo de saída (stdout se não especificado)")
    parser.add_argument("--update-main", action="store_true",
                        help="Atualiza diretamente o arquivo main/filter_coefficients.h")

    args = parser.parse_args()

    # --fir-taps é um atalho que define ambos
    if args.fir_taps is not None:
        fir_f32_taps = args.fir_taps
        fir_q15_taps = args.fir_taps
    else:
        fir_f32_taps = args.fir_f32_taps
        fir_q15_taps = args.fir_q15_taps

    cutoff = args.cutoff
    if args.type in ["lowpass", "highpass"] and len(cutoff) > 1:
        cutoff = cutoff[0]
    elif args.type in ["bandstop", "bandpass"] and len(cutoff) == 1:
        parser.error(f"O tipo '{args.type}' requer duas frequências de corte (ex: --cutoff 600 1100)")

    print(f"Projetando filtro {args.type.upper()} @ Fs = {args.fs} Hz...", file=sys.stderr)
    print(f"  Frequência(s) de corte : {cutoff} Hz",  file=sys.stderr)
    print(f"  FIR Float32 Taps       : {fir_f32_taps}", file=sys.stderr)
    print(f"  FIR Q15     Taps       : {fir_q15_taps}", file=sys.stderr)
    print(f"  Estágios IIR           : {args.iir_stages}", file=sys.stderr)

    # --- Geração independente por seção ---
    fir_f32 = design_fir_f32(numtaps=fir_f32_taps, cutoff=cutoff,
                              fs=args.fs, pass_zero=args.type)

    fir_q15 = design_fir_q15(numtaps=fir_q15_taps, cutoff=cutoff,
                              fs=args.fs, pass_zero=args.type)

    iir_f32, iir_q15, actual_stages = design_iir(
        stages=args.iir_stages, cutoff=cutoff,
        fs=args.fs, btype=args.type
    )

    print(f"  IIR estágios reais     : {actual_stages}", file=sys.stderr)

    c_code = format_c_code(fir_f32, fir_q15, iir_f32, iir_q15,
                           sample_rate=args.fs)

    if args.update_main:
        update_header_file(c_code)
    elif args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(c_code)
        print(f"Código C gerado com sucesso em '{args.output}'!", file=sys.stderr)
    else:
        print(c_code)


if __name__ == "__main__":
    main()
