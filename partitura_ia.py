"""
Reconhecimento e Transposição de Partituras com IA (OMR)
=========================================================
Pipeline: Imagem JPG → OpenCV → oemer (PyTorch) → music21 → Transposição → MusicXML / MIDI / PNG

Partitura: "Viva o Carnaval" — Chiquinha Gonzaga (Musica Brasilis)
Autor: lucaseduardo79
"""

import os
import glob
import subprocess
import time
from io import BytesIO

import requests
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')   # sem display — salva em arquivo
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image as PILImage

import torch
from music21 import (
    converter, stream, note, chord, interval as m21interval,
    key, tempo, meter, metadata, pitch as m21pitch, environment
)

os.makedirs('output', exist_ok=True)

# ─── Configuração ─────────────────────────────────────────────────────────────
URL_PARTITURA = None   # None = usa arquivo local
ARQUIVO_LOCAL = "Tarantella_napoletana.jpeg"
INTERVALO     = 'P4'   # Quarta Justa (+5 st): Dó → Fá | troque por 'P5', '-M2', 'M3' etc.
# ──────────────────────────────────────────────────────────────────────────────


# ── 1. GPU ────────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[GPU] Dispositivo: {device}")
if torch.cuda.is_available():
    print(f"[GPU] {torch.cuda.get_device_name(0)}")


# ── 2. Carrega a partitura ────────────────────────────────────────────────────
if URL_PARTITURA:
    print("\n[1/6] Baixando partitura...")
    response = requests.get(URL_PARTITURA, timeout=30)
    response.raise_for_status()
    with open(ARQUIVO_LOCAL, 'wb') as f:
        f.write(response.content)
    print(f"      ✅ {ARQUIVO_LOCAL} ({len(response.content)/1024:.1f} KB)")
else:
    print(f"\n[1/6] Usando arquivo local: {ARQUIVO_LOCAL}")
    if not os.path.exists(ARQUIVO_LOCAL):
        raise FileNotFoundError(f"Arquivo não encontrado: {ARQUIVO_LOCAL}")

img_original = PILImage.open(ARQUIVO_LOCAL)
print(f"      Dimensões: {img_original.size[0]}×{img_original.size[1]} | Modo: {img_original.mode}")


# ── 3. Pré-processamento com OpenCV ──────────────────────────────────────────
print("\n[2/6] Pré-processamento com OpenCV...")

img_bgr  = cv2.imread(ARQUIVO_LOCAL)
img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

img_bin = cv2.adaptiveThreshold(
    img_gray, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY, 21, 10
)
kernel_h   = cv2.getStructuringElement(cv2.MORPH_RECT, (img_gray.shape[1] // 10, 1))
img_pautas = cv2.morphologyEx(~img_bin, cv2.MORPH_OPEN, kernel_h)
img_notas  = cv2.subtract(~img_bin, img_pautas)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Pipeline de Pré-processamento da Partitura', fontsize=14, fontweight='bold')
axes[0, 0].imshow(img_rgb);    axes[0, 0].set_title('1. Original (RGB)');         axes[0, 0].axis('off')
axes[0, 1].imshow(img_gray, cmap='gray'); axes[0, 1].set_title('2. Escala de Cinza'); axes[0, 1].axis('off')
axes[1, 0].imshow(img_bin,  cmap='gray'); axes[1, 0].set_title('3. Binarização Adaptativa'); axes[1, 0].axis('off')
axes[1, 1].imshow(img_notas,cmap='gray'); axes[1, 1].set_title('4. Notas Isoladas');  axes[1, 1].axis('off')
plt.tight_layout()
plt.savefig('output/preprocessing.png', dpi=120)
plt.close()
print("      ✅ output/preprocessing.png")


# ── 4. OMR com oemer ─────────────────────────────────────────────────────────
print("\n[3/6] OMR com oemer (PyTorch)...")
xml_path = None
start = time.time()

# Tenta API Python (oemer >= 0.1.7 expõe via oemer.ete.extract)
def _oemer_python(img_path):
    from oemer.ete import extract
    import argparse
    # extract() espera o Namespace do argparse — montamos manualmente
    args = argparse.Namespace(
        img_path=img_path,
        output_path='output',
        use_tf=False,
        without_deskew=False,
        save_cache=False,
        debug=False,
    )
    extract(args)

# oemer salva como output/<nome_completo_incluindo_extensao>.musicxml
# ex: Tarantella_napoletana.jpeg → output/Tarantella_napoletana.jpeg.musicxml
xml_omr = f'output/{os.path.basename(ARQUIVO_LOCAL)}.musicxml'

try:
    _oemer_python(ARQUIVO_LOCAL)
    if not os.path.exists(xml_omr):
        raise FileNotFoundError(f"XML esperado não encontrado: {xml_omr}")
    xml_path = xml_omr
    print(f"      ✅ OMR (API Python) em {time.time()-start:.1f}s → {xml_path}")
except Exception as e_api:
    print(f"      ⚠️  API Python falhou ({type(e_api).__name__}: {e_api}), tentando CLI...")
    try:
        result = subprocess.run(
            ['oemer', ARQUIVO_LOCAL, '-o', 'output'],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr[:400])
        if not os.path.exists(xml_omr):
            raise FileNotFoundError(f"XML esperado não encontrado: {xml_omr}")
        xml_path = xml_omr
        print(f"      ✅ OMR (CLI) em {time.time()-start:.1f}s → {xml_path}")
    except Exception as e_cli:
        print(f"      ⚠️  oemer CLI falhou: {e_cli}")
        print("      📝 Usando melodia de demonstração (music21)...")


# ── 5. Carrega / cria a partitura ─────────────────────────────────────────────
print("\n[4/6] Carregando partitura com music21...")

if xml_path and os.path.exists(xml_path):
    score  = converter.parse(xml_path)
    origem = f"OMR — {xml_path}"
else:
    # Melodia de demonstração estilo polca/choro — seção A de "Viva o Carnaval"
    p = stream.Part()
    p.append(meter.TimeSignature('2/4'))
    p.append(tempo.MetronomeMark(number=120))
    p.append(key.KeySignature(0))
    melodia = [
        ('E5',.5),('D5',.5), ('C5',.5),('B4',.5),
        ('A4',.5),('G4',.25),('A4',.25), ('B4',1.0),
        ('G5',.5),('F5',.5), ('E5',.5),('D5',.5),
        ('C5',.5),('E5',.25),('D5',.25), ('C5',1.0),
        ('A4',.5),('B4',.5), ('C5',.5),('D5',.5),
        ('E5',.5),('G5',.5), ('F5',.5),('E5',.5),
        ('D5',.5),('C5',.25),('D5',.25), ('E5',.5),('G4',.5),
        ('A4',.5),('B4',.5), ('C5',1.0),
    ]
    for pitch_name, dur in melodia:
        n = note.Note(pitch_name)
        n.duration.quarterLength = dur
        p.append(n)
    score = stream.Score([p])
    score.metadata = metadata.Metadata()
    score.metadata.title    = 'Viva o Carnaval (demonstração)'
    score.metadata.composer = 'Chiquinha Gonzaga'
    origem = "Demonstração (music21)"

print(f"      Origem: {origem}")


# ── 5a. Análise da partitura ──────────────────────────────────────────────────
try:
    k = score.analyze('key')
    print(f"      Tonalidade: {k.tonic.name} {k.mode} (correlação {k.correlationCoefficient:.4f})")
except Exception:
    k = key.Key('C')
    print(f"      Tonalidade: Dó Maior (manual)")

notas_obj = score.flatten().notes
print(f"      Notas/acordes: {len(notas_obj)}")

# Histograma de notas
pitch_counter = {}
for n in notas_obj:
    pitches = n.pitches if isinstance(n, chord.Chord) else [n.pitch]
    for p in pitches:
        pitch_counter[p.name] = pitch_counter.get(p.name, 0) + 1

if pitch_counter:
    fig, ax = plt.subplots(figsize=(10, 4))
    nomes    = list(pitch_counter.keys())
    contagens = list(pitch_counter.values())
    bars = ax.bar(nomes, contagens, color='steelblue', edgecolor='navy')
    ax.set_title('Frequência das Notas — Partitura Original', fontsize=12, fontweight='bold')
    ax.set_xlabel('Nota'); ax.set_ylabel('Ocorrências')
    for bar, c in zip(bars, contagens):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + .1,
                str(c), ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig('output/histograma_notas.png', dpi=120)
    plt.close()
    print("      ✅ output/histograma_notas.png")


def _piano_roll(ax, score_obj, titulo, cor_nota, cor_borda):
    for n in score_obj.flatten().notes:
        if not isinstance(n, note.Note):
            continue
        midi  = n.pitch.midi
        onset = float(n.offset)
        dur   = float(n.duration.quarterLength)
        rect  = mpatches.FancyBboxPatch(
            (onset, midi - .4), max(dur * .88, .1), .8,
            boxstyle='round,pad=0.05', linewidth=.5,
            edgecolor=cor_borda, facecolor=cor_nota, alpha=.85
        )
        ax.add_patch(rect)
        ax.text(onset + dur / 2, midi, n.pitch.name,
                ha='center', va='center', fontsize=6, color='white', fontweight='bold')
    ax.autoscale()
    ax.set_title(titulo, fontsize=12, fontweight='bold')
    ax.set_xlabel('Tempo (quartos)'); ax.set_ylabel('Nota MIDI')
    midi_vals = sorted({n.pitch.midi for n in score_obj.flatten().notes if isinstance(n, note.Note)})
    if midi_vals:
        ax.set_yticks(midi_vals)
        ax.set_yticklabels([m21pitch.Pitch(m).nameWithOctave for m in midi_vals], fontsize=7)
    ax.grid(axis='x', alpha=.3)


# ── 6. Transposição ───────────────────────────────────────────────────────────
print(f"\n[5/6] Transposição ({INTERVALO})...")
intervalo       = m21interval.Interval(INTERVALO)
score_transposto = score.transpose(intervalo)

try:
    k_novo = score_transposto.analyze('key')
except Exception:
    k_novo = key.Key('F')

print(f"      {k.tonic.name} {k.mode}  →  {k_novo.tonic.name} {k_novo.mode}  ({intervalo.niceName}, {intervalo.semitones:+d} st)")

# Piano roll comparativo
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
fig.suptitle('Piano Roll — Original vs Transposta', fontsize=14, fontweight='bold')
_piano_roll(ax1, score,           f'Original — {k.tonic.name} {k.mode}',        'steelblue', 'navy')
_piano_roll(ax2, score_transposto, f'Transposta — {k_novo.tonic.name} {k_novo.mode} ({intervalo.niceName})', 'tomato', 'darkred')
plt.tight_layout()
plt.savefig('output/piano_roll.png', dpi=120)
plt.close()
print("      ✅ output/piano_roll.png")


# ── 7. Exportação ─────────────────────────────────────────────────────────────
print("\n[6/6] Exportando partitura transposta...")

score_transposto.metadata = metadata.Metadata()
score_transposto.metadata.title    = f'Viva o Carnaval — Transposta ({intervalo.niceName})'
score_transposto.metadata.composer = 'Chiquinha Gonzaga'

path_xml  = 'output/partitura_transposta.musicxml'
path_midi = 'output/partitura_transposta.mid'
path_png  = 'output/partitura_transposta.png'

score_transposto.write('musicxml', path_xml)
print(f"      ✅ MusicXML → {path_xml}")

score_transposto.write('midi', path_midi)
print(f"      ✅ MIDI     → {path_midi}")

try:
    score_transposto.write('musicxml.png', path_png)
    # music21/MuseScore gera sufixo de página: partitura_transposta-1.png, -2.png...
    pngs = sorted(glob.glob('output/partitura_transposta*.png'))
    if pngs:
        path_png = pngs[0]
    print(f"      ✅ PNG      → {path_png}")
except Exception as e:
    print(f"      ⚠️  PNG (MuseScore ausente): {e}")


# ── Resumo ────────────────────────────────────────────────────────────────────
print()
print("╔══════════════════════════════════════════════════════════╗")
print("║          RESUMO — Partitura IA (lucaseduardo79)          ║")
print("╠══════════════════════════════════════════════════════════╣")
peca = os.path.splitext(os.path.splitext(os.path.basename(ARQUIVO_LOCAL))[0])[0].replace('_', ' ')
print(f"║  Peça           : {peca:<42}║")
print(f"║  Fonte          : {origem[:42]:<42}║")
print(f"║  Tom original   : {(k.tonic.name + ' ' + k.mode):<42}║")
print(f"║  Tom transposto : {(k_novo.tonic.name + ' ' + k_novo.mode):<42}║")
print(f"║  Intervalo      : {(intervalo.niceName + ' (' + INTERVALO + ')'):<42}║")
print(f"║  Notas          : {len(notas_obj):<42}║")
print("╠══════════════════════════════════════════════════════════╣")

arquivos = [
    ('preprocessing.png',          'output/preprocessing.png'),
    ('histograma_notas.png',       'output/histograma_notas.png'),
    ('piano_roll.png',             'output/piano_roll.png'),
    ('partitura_transposta.musicxml', path_xml),
    ('partitura_transposta.mid',   path_midi),
    ('partitura_transposta.png',   path_png),
]
for nome, caminho in arquivos:
    existe  = os.path.exists(caminho)
    status  = '✅' if existe else '⚠️ '
    tamanho = f"{os.path.getsize(caminho)/1024:.1f}KB" if existe else 'não gerado'
    print(f"║  {status} {nome:<36} {tamanho:>8} ║")

print("╚══════════════════════════════════════════════════════════╝")
