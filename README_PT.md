# NVS Renderer

| [English](README.md) | Português |

Um renderizador OpenGL para modelos PyTorch de 3D Novel View Synthesis (NVS).

Por enquanto, consegue renderizar cenas do [VDST](https://github.com/gammag4/vdst).

Cena renderizada:

https://github.com/user-attachments/assets/35a33c5e-fd6a-473b-b55a-ea6f4a667a93

Imagens-fonte:

![Imagens-fonte](results/sources.png)

<!-- lvsm -->
<!-- https://github.com/user-attachments/assets/16de9309-e82a-4c30-a4e0-e8285eb4e954

https://github.com/user-attachments/assets/7071a7dc-cba4-410b-ab48-004afeadcf46 -->

## Uso

Instale os requisitos:

- Alguma distribuição conda (recomendamos usar [Miniforge](https://conda-forge.org/download/))
- NVIDIA drivers com suporte para CUDA >= 13.0

Clone esse repositório:

```bash
git clone https://github.com/gammag4/nvs_renderer
cd nvs_renderer
```

Crie o ambiente conda:

```bash
conda create -n nvs_renderer python=3.13
conda activate nvs_renderer
conda install -c conda-forge ffmpeg
pip install -r requirements.txt
```

Clone [VDST](https://github.com/gammag4/vdst) para a pasta `VDST` e siga as instruções para configurar o modelo para inferência, usando o mesmo ambiente conda.

Faça a build e rode:

```bash
python render.py --module renderers/vdst.py
```

Os controles são WASD para frente, esquerda, trás, direita, ctrl esquerdo/espaço para baixo/cima, mouse para movimentação de câmera e T para alternar mapa de distância se houver um.
Pressione ESC uma vez para destravar o mouse e pressione duas para fechar.

### Usando com outros modelos

#### Como um módulo

Para renderizar usando outro modelo, importe e use a função `render_model` com o seguinte formato:

```py
render_model(n_frames, initial_T, render, device, render_resolution, window_resolution=(800, 800))
```

Onde:

- `n_frames: int`: Número de frames na cena (deve ser 1 no caso de cenas estáticas)
- `initial_T: tensor`: Matriz 4x4 de transformação da câmera inicial
- `render(T: tensor, frame_index: int) -> I: (tensor, tensor)`: Uma função que recebe a matriz 4x4 de transformação de câmera e o índice do frame atual (que vai ser sempre zero em NVS estática) e retorna uma tupla (image, depth) com a imagem renderizada (shape `(C=3, H, W)`) e o mapa de distâncias renderizado como float em metros (ou None se não tem estimação de profundidade) (shape `(H, W)`) naquela posição
- `device: str`: Qual dispositivo usar (deve ser um dispositivo CUDA)
- `render_resolution: (int, int)`: Qual resolução usar para renderizar imagens (deve ter o mesmo shape que a saída de `render(T)`)
- `window_resolution: (int, int)`: (opcional) Qual resolução usar para a janela

#### Como um script

Crie um módulo similar ao `renderers/vdst.py` que exporta todos os parâmetros descritos na seção anterior.

Então faça as configurações usuais, faça a build e rode:

```bash
python render.py --module <caminho_para_o_seu_modulo>
```
