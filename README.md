# DIO---Criando-um-Sistema-de-Reconhecimento-Facial-do-Zero
### Parte 1: Detecção Facial

Para a detecção facial, usaremos o modelo Haar Cascades com OpenCV, conforme seu exemplo. Isso é simples e eficiente para começar.

### Parte 2: Classificação Facial

Para a classificação facial, usaremos a biblioteca face_recognition que é baseada no modelo dlib e muito eficiente para reconhecimento facial.

### Explicação do Código


    Carregar Faces Conhecidas:
        A função load_known_faces carrega imagens de um diretório e gera os embeddings das faces conhecidas, associando-os aos nomes dos indivíduos.

    Captura de Vídeo:
        Usamos OpenCV para capturar o vídeo da webcam ou de um arquivo de vídeo.

    Detecção e Reconhecimento de Faces:
        Cada frame é processado para detectar as localizações das faces e calcular os embeddings.
        Os embeddings das faces detectadas são comparados com os embeddings das faces conhecidas para identificar as pessoas.
        Desenhamos um retângulo ao redor das faces detectadas e escrevemos o nome da pessoa identificada abaixo da face.

### Requisitos

    Instale as bibliotecas necessárias:

    cmake
    pip install opencv-python face_recognition numpy

### Teste

Execute o script e aponte a webcam para diferentes pessoas ou use um vídeo para verificar a detecção e reconhecimento facial.

Este código cobre tanto a detecção facial quanto o reconhecimento, usando modelos pré-treinados para simplificar o processo.