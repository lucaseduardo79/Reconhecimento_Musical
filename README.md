# 🎵 Reconhecimento Musical com IA

Pipeline completo de **Optical Music Recognition (OMR)** com transposição automática e geração de partitura, rodando em container Docker com suporte a GPU CUDA.

> **Partitura de exemplo:** *Tarantella Napoletana* — imagem JPEG de entrada, MusicXML + MIDI + PNG de saída.

---

## 📋 Índice

1. [O que o projeto faz](#o-que-o-projeto-faz)
2. [Pipeline completo](#pipeline-completo)
3. [Estrutura de arquivos](#estrutura-de-arquivos)
4. [Pré-requisitos](#pré-requisitos)
5. [Como executar](#como-executar)
6. [Escolhas de desenvolvimento](#escolhas-de-desenvolvimento)
7. [Funções e módulos](#funções-e-módulos)
8. [Saídas geradas](#saídas-geradas)
9. [Configuração e personalização](#configuração-e-personalização)
10. [Limitações conhecidas](#limitações-conhecidas)

---

## O que o projeto faz

Dado uma **imagem de partitura** (JPEG/PNG), o sistema:

1. **Pré-processa** a imagem com OpenCV para realçar símbolos musicais
2. **Reconhece** automaticamente as notas usando um modelo de deep learning (oemer)
3. **Analisa** a tonalidade, distribuição de notas e estrutura rítmica
4. **Transpõe** a peça para outro tom configurável
5. **Exporta** o resultado em MusicXML, MIDI e PNG de partitura renderizada

Tudo isso encapsulado em um único comando Docker — sem instalação manual de dependências.

---

## Pipeline completo

```
┌─────────────────────────────────────────────────────────────────┐
│                    PIPELINE PARTITURA IA                        │
├─────────────┬───────────────────────────────────────────────────┤
│  ENTRADA    │  Imagem JPEG/PNG de partitura                     │
├─────────────┼───────────────────────────────────────────────────┤
│  ETAPA 1    │  Pré-processamento (OpenCV)                       │
│             │  • Conversão RGB → Cinza                          │
│             │  • Binarização adaptativa (Gaussian)              │
│             │  • Extração morfológica de pautas                 │
│             │  • Isolamento das cabeças de nota                 │
├─────────────┼───────────────────────────────────────────────────┤
│  ETAPA 2    │  OMR — Optical Music Recognition (oemer)          │
│             │  • UNet grande: segmentação de pautas e símbolos  │
│             │  • UNet pequena: classificação de camadas         │
│             │  • SVM: classificação de símbolos musicais        │
│             │  • Saída: arquivo MusicXML                        │
├─────────────┼───────────────────────────────────────────────────┤
│  ETAPA 3    │  Análise musical (music21)                        │
│             │  • Detecção de tonalidade (Krumhansl-Schmuckler)  │
│             │  • Histograma de alturas (pitch class)            │
│             │  • Piano roll interativo                          │
├─────────────┼───────────────────────────────────────────────────┤
│  ETAPA 4    │  Transposição (music21)                           │
│             │  • Intervalo configurável (P4, P5, M3, -M2...)    │
│             │  • Preserva duração, dinâmica e articulação       │
├─────────────┼───────────────────────────────────────────────────┤
│  SAÍDA      │  MusicXML  •  MIDI  •  PNG  •  Gráficos          │
└─────────────┴───────────────────────────────────────────────────┘
```

---

## Estrutura de arquivos

```
Reconhecimento_Musical/
│
├── partitura_ia.py              # Script principal (pipeline completo)
├── Tarantella_napoletana.jpeg   # Partitura de exemplo
│
├── Dockerfile                   # Imagem Docker com CUDA + MuseScore
├── docker-compose.yml           # Orquestração com GPU e volumes
├── requirements.txt             # Dependências Python
├── start.sh                     # Entrypoint: download de modelos + execução
│
└── output/                      # Gerado automaticamente
    ├── preprocessing.png        # Visualização do pré-processamento
    ├── histograma_notas.png     # Distribuição de alturas
    ├── piano_roll.png           # Piano roll: original vs transposta
    ├── Tarantella_napoletana.musicxml  # OMR bruto
    ├── partitura_transposta.musicxml   # Após transposição
    ├── partitura_transposta.mid        # MIDI para reprodução
    └── partitura_transposta-1.png      # Partitura renderizada (MuseScore)
```

---

## Pré-requisitos

| Requisito | Versão mínima | Observação |
|-----------|--------------|------------|
| Docker Desktop | 4.x | Com suporte a GPU habilitado |
| NVIDIA Driver | 525+ | CUDA 12.x no host |
| NVIDIA Container Toolkit | 1.14+ | Para `--gpus` no Docker |
| VRAM | 4 GB | O modelo UNet usa ~1.5 GB |

### Habilitando GPU no Docker Desktop (Windows)

`Docker Desktop → Settings → Resources → GPU → ✅ Enable GPU`

Verifique com:
```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

---

## Como executar

```bash
# Clone ou navegue até a pasta
cd Reconhecimento_Musical

# Primeira execução (build da imagem + download dos modelos ~400 MB)
docker compose up --build

# Execuções subsequentes (modelos já no volume, muito mais rápido)
docker compose up
```

Os resultados aparecem em `output/` em tempo real durante a execução.

### Trocar a partitura de entrada

Edite as duas primeiras constantes em `partitura_ia.py`:

```python
URL_PARTITURA = None                     # None = usa arquivo local
ARQUIVO_LOCAL = "minha_partitura.jpeg"   # coloque o arquivo na mesma pasta
```

Para usar uma URL remota:
```python
URL_PARTITURA = "https://exemplo.com/partitura.jpg"
ARQUIVO_LOCAL = "output/partitura_baixada.jpg"
```

---

## Escolhas de desenvolvimento

### Por que Docker?

A stack combina CUDA, ONNX Runtime, PyTorch, MuseScore e bibliotecas C (OpenCV, libgl) — dependências que conflitam facilmente entre si em instalações locais. O Docker garante que qualquer máquina com GPU NVIDIA reproduz exatamente o mesmo ambiente, eliminando o "funciona na minha máquina".

A imagem base escolhida foi `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime` porque já traz PyTorch, CUDA toolkit e cuDNN pré-compilados, reduzindo o tamanho final e o tempo de build em relação a partir de uma imagem Ubuntu pura.

### Por que oemer para o OMR?

O campo de Optical Music Recognition tem soluções como **Audiveris** (Java, regras manuais), **SheetVision** e abordagens baseadas em YOLO puro. O **oemer** foi escolhido porque:

- É **end-to-end em Python/PyTorch**, sem dependências Java
- Usa uma arquitetura **U-Net** para segmentação semântica de símbolos musicais, mais robusta que detecção de objetos simples para partituras densas
- Exporta diretamente para **MusicXML**, o formato padrão da indústria
- Tem a melhor relação custo-benefício entre precisão e facilidade de uso em 2024–2025

**Limitação aceita:** os modelos SVM embutidos no oemer foram treinados com scikit-learn 1.2, gerando `InconsistentVersionWarning` com versões mais recentes. Não afeta o resultado prático — é um aviso de compatibilidade de serialização.

### Por que music21?

O **music21** do MIT é a biblioteca de referência para musicologia computacional em Python. Alternativas como **pretty_midi** são mais simples mas tratam música apenas como eventos MIDI sem compreensão de teoria musical. O music21 entende conceitos como tonalidade, intervalos, compassos e armaduras de clave — essencial para transposição semanticamente correta (não apenas shift de MIDI numbers).

A detecção de tonalidade usa o algoritmo **Krumhansl-Schmuckler**, que correlaciona a distribuição de alturas da peça com perfis de tonalidades maiores e menores — o mesmo método usado em pesquisa musicológica.

### Por que `QT_QPA_PLATFORM=offscreen`?

O MuseScore usa Qt para renderização. Em containers sem servidor X11 real, o plugin `xcb` (padrão) tenta conectar a um display que não existe. A variável `QT_QPA_PLATFORM=offscreen` instrui o Qt a usar um framebuffer em memória para renderização, sem nenhuma janela real — equivalente a um "modo headless" nativo. É mais limpo e confiável do que manter um servidor Xvfb rodando em segundo plano.

### Por que volume para os checkpoints?

Os modelos ONNX do oemer somam ~400 MB e são baixados na primeira execução. Sem um volume persistente, seriam re-baixados a cada `docker compose up`. O volume nomeado `oemer-checkpoints` monta sobre o diretório de instalação do pacote, tornando os pesos persistentes entre execuções sem necessidade de incluí-los na imagem (o que aumentaria o tamanho do build para todos os usuários).

### Binarização adaptativa vs global

A imagem de uma partitura digitalizada frequentemente tem iluminação irregular (sombras de dobras, margens amareladas). A binarização **global** (Otsu) falha nesse cenário porque escolhe um único limiar para toda a imagem. A binarização **adaptativa gaussiana** calcula um limiar local para cada região, sendo muito mais robusta para documentos históricos e digitalizações de baixa qualidade.

---

## Funções e módulos

### `partitura_ia.py`

#### Constantes de configuração (topo do arquivo)
```python
URL_PARTITURA = None          # URL remota ou None para arquivo local
ARQUIVO_LOCAL = "..."         # Caminho da imagem de entrada
INTERVALO     = 'P4'          # Intervalo de transposição (notação music21)
```

Intervalos disponíveis:

| Código | Nome | Semitons | Exemplo |
|--------|------|----------|---------|
| `'P4'` | Quarta Justa | +5 | Dó → Fá |
| `'P5'` | Quinta Justa | +7 | Dó → Sol |
| `'M3'` | Terça Maior | +4 | Dó → Mi |
| `'-M2'` | Segunda Maior ↓ | -2 | Dó → Si♭ |
| `'-P5'` | Quinta Justa ↓ | -7 | Dó → Fá (oitava abaixo) |

---

#### Bloco 1 — Carregamento da imagem
Baixa via HTTP ou lê localmente, valida existência do arquivo e imprime dimensões. A separação entre URL e arquivo local permite usar o mesmo script tanto em modo automático (pipeline CI/CD com URL fixa) quanto interativo (arquivo local).

---

#### Bloco 2 — Pré-processamento OpenCV

```python
img_bin = cv2.adaptiveThreshold(img_gray, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10)
```

- **`blockSize=21`** — janela de 21×21 pixels para cálculo do limiar local. Calibrado para pautas musicais em resoluções típicas de scan (150–300 DPI)
- **`C=10`** — constante subtraída do limiar para aumentar a sensibilidade a pixels escuros (notas e linhas)

```python
kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (largura//10, 1))
img_pautas = cv2.morphologyEx(~img_bin, cv2.MORPH_OPEN, kernel_h)
```

O kernel horizontal estreito (1px de altura, 10% da largura) extrai apenas linhas longas e contínuas — as pautas musicais. A operação morfológica `OPEN` (erosão seguida de dilatação) remove objetos menores que o kernel, deixando apenas as linhas.

Saída: `output/preprocessing.png` com as 4 etapas lado a lado.

---

#### Bloco 3 — OMR com oemer

```python
def _oemer_python(img_path):
    from oemer.ete import extract
    args = argparse.Namespace(
        img_path=img_path,
        output_path='output',
        use_tf=False,
        without_deskew=False,
        save_cache=False,
        debug=False,
    )
    extract(args)
```

A função `extract()` do oemer espera um `argparse.Namespace` porque foi projetada como API interna do CLI. Montamos o Namespace manualmente para evitar chamar o subprocess — ganhamos acesso direto ao progresso e exceções Python.

O script detecta automaticamente o nome do arquivo gerado pelo oemer:
```python
base_name = os.path.splitext(os.path.basename(ARQUIVO_LOCAL))[0]
xml_omr   = f'output/{base_name}.musicxml'
```
Isso evita ambiguidade com arquivos de execuções anteriores.

Fallback: se a API Python falhar, tenta o CLI via `subprocess.run(['oemer', ...])`.

---

#### Bloco 4 — Análise musical

```python
k = score.analyze('key')   # Krumhansl-Schmuckler
```

Gera dois gráficos:
- **Histograma de alturas** — frequência de cada nota (C, D, E...) independente de oitava
- **Piano roll** — representação temporal das notas: eixo X = tempo em quartos, eixo Y = altura MIDI

O piano roll usa `matplotlib.patches.FancyBboxPatch` para cada nota, com cor e espessura proporcional à duração.

---

#### Bloco 5 — Transposição

```python
score_transposto = score.transpose(m21interval.Interval(INTERVALO))
```

O `music21` transpõe a partitura inteira preservando:
- Relações intervalares entre vozes
- Armadura de clave atualizada
- Articulações e dinâmicas

Gera um piano roll comparativo (original em azul / transposta em vermelho).

---

#### Bloco 6 — Exportação

| Formato | Método music21 | Uso |
|---------|---------------|-----|
| MusicXML | `write('musicxml', ...)` | Edição no MuseScore/Sibelius/Finale |
| MIDI | `write('midi', ...)` | Reprodução em DAW ou player |
| PNG | `write('musicxml.png', ...)` | Visualização/impressão |

O PNG é gerado via MuseScore em modo offscreen (`QT_QPA_PLATFORM=offscreen`). O music21 invoca o MuseScore internamente e retorna um arquivo por página com sufixo `-1.png`, `-2.png` etc.

---

### `start.sh`

Script de entrypoint do container. Responsabilidades:
1. Imprimir diagnóstico de GPU (PyTorch)
2. Verificar se os checkpoints do oemer já foram baixados (via existência de `checkpoints/unet_big/`)
3. Baixar os modelos apenas se ausentes (`from oemer.download import download`)
4. Executar o script principal

### `Dockerfile`

| Instrução | Justificativa |
|-----------|--------------|
| `FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime` | CUDA + cuDNN + PyTorch já compilados |
| `musescore3` via apt | Renderização de partitura → PNG sem interface gráfica |
| `ENV QT_QPA_PLATFORM=offscreen` | Qt headless sem Xvfb |
| `sed -i 's/\r//'` no start.sh | Remove CRLF gerado no Windows antes de executar no Linux |
| music21 configurado via `python3 -c` inline | Evita quebra de parse do Dockerfile com `FROM` falso em strings multiline |

### `docker-compose.yml`

```yaml
volumes:
  - .:/workspace                           # hot-reload: editar localmente reflete no container
  - oemer-checkpoints:/opt/conda/.../checkpoints  # modelos persistem entre reinicializações
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          capabilities: [gpu]             # Docker Compose v2 — sem necessidade de --gpus
```

---

## Saídas geradas

| Arquivo | Descrição |
|---------|-----------|
| `output/preprocessing.png` | 4 etapas do processamento de imagem lado a lado |
| `output/histograma_notas.png` | Frequência de cada classe de nota na partitura |
| `output/piano_roll.png` | Representação temporal: original (azul) vs transposta (vermelho) |
| `output/<nome_entrada>.musicxml` | MusicXML gerado pelo OMR (bruto) |
| `output/partitura_transposta.musicxml` | MusicXML final após transposição |
| `output/partitura_transposta.mid` | MIDI para reprodução direta |
| `output/partitura_transposta-1.png` | Partitura renderizada pelo MuseScore (página 1) |

---

## Configuração e personalização

### Mudar o intervalo de transposição

Em `partitura_ia.py`, linha 12:
```python
INTERVALO = 'P5'   # Quinta Justa: Lá menor → Mi menor
```

### Usar URL remota

```python
URL_PARTITURA = "https://musicabrasilis.org.br/media/images/cg_viva_o_carnaval.original.jpg"
ARQUIVO_LOCAL = "output/partitura_baixada.jpg"
```

### Rodar sem GPU

O script detecta automaticamente e usa CPU se CUDA não estiver disponível. Retire o bloco `deploy.resources` do `docker-compose.yml` e remova `--gpus` se precisar rodar em máquina sem NVIDIA.

---

## Limitações conhecidas

| Limitação | Causa | Contorno |
|-----------|-------|----------|
| `InconsistentVersionWarning` (sklearn) | oemer foi treinado com sklearn 1.2, container usa 1.8 | Apenas aviso — não afeta resultado |
| `ConvTranspose fallback to CPU` | ONNX Runtime não suporta padding assimétrico em CUDA para esse op | Apenas performance — resultado idêntico |
| PNG com sufixo `-1.png` | music21/MuseScore gera um arquivo por página | O script detecta automaticamente via glob |
| OMR impreciso em partituras manuscritas | oemer foi treinado em partituras impressas | Use imagens de boa qualidade e impressas |
| Primeira execução demora ~7 min | Download de modelos (~400 MB) + build Docker | Execuções seguintes são ~5 min |

---

## Tecnologias utilizadas

| Biblioteca | Versão | Função |
|------------|--------|--------|
| PyTorch | 2.5.1 | Backend de deep learning do oemer |
| ONNX Runtime | — | Inferência otimizada dos modelos U-Net |
| oemer | ≥0.1.7 | Optical Music Recognition end-to-end |
| music21 | ≥9.1 | Análise, transposição e exportação musical |
| OpenCV | ≥4.9 | Pré-processamento de imagem |
| MuseScore 3 | apt | Renderização de partitura → PNG |
| matplotlib | ≥3.8 | Piano roll e histogramas |
| CUDA | 12.4 | Aceleração GPU (compatível com drivers 12.x) |

---

## Autor

**lucaseduardo79** — [github.com/lucaseduardo79](https://github.com/lucaseduardo79)
