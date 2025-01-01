import cv2
from deepface import DeepFace
import os
import uuid

# Função para criar diretórios
def create_dir_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Função para salvar a imagem do rosto
def save_face(image, directory, name):
    filename = f"{name}.jpg"
    filepath = os.path.join(directory, filename)
    cv2.imwrite(filepath, image)
    print(f"Rosto salvo em: {filepath}")

# Diretórios para salvar rostos conhecidos e desconhecidos
known_faces_dir = 'known_faces'
unknown_faces_dir = 'unknown_faces'
create_dir_if_not_exists(known_faces_dir)
create_dir_if_not_exists(unknown_faces_dir)

# Função para carregar as imagens de rostos conhecidos e retornar embeddings e nomes
def load_known_faces(known_faces_dir):
    known_face_encodings = []
    known_face_names = []

    for name in os.listdir(known_faces_dir):
        person_dir = os.path.join(known_faces_dir, name)
        if not os.path.isdir(person_dir):
            continue
        for filename in os.listdir(person_dir):
            filepath = os.path.join(person_dir, filename)
            image = cv2.imread(filepath)
            if image is None:
                print(f"Erro ao carregar imagem {filepath}")
                continue
            print(f"Carregando imagem {filepath}")
            # Calcular embeddings usando o modelo FaceNet
            embedding = DeepFace.represent(img_path=filepath, model_name='Facenet')[0]['embedding']
            known_face_encodings.append(embedding)
            known_face_names.append(name)
    return known_face_encodings, known_face_names

# Carrega as faces conhecidas e seus nomes
known_face_encodings, known_face_names = load_known_faces(known_faces_dir)

# Verifica se as faces conhecidas foram carregadas corretamente
print("Faces conhecidas carregadas:")
for name, encoding in zip(known_face_names, known_face_encodings):
    print(f"Nome: {name}, Embedding shape: {len(encoding)}")

# Inicia a captura de vídeo (use 0 para webcam ou 'path/to/video.mp4' para um vídeo)
video_capture = cv2.VideoCapture('video/01.mp4')  # Use 0 para webcam ou 'path/to/video.mp4' para um vídeo

# Carrega o classificador Haar para detecção de rostos
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Converte o frame para escala de cinza
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detecta a localização das faces no frame
    face_locations = face_cascade.detectMultiScale(
        gray_frame,
        scaleFactor=1.1,
        minNeighbors=17,  # Aumentar este valor ajuda a reduzir falsos positivos
        minSize=(30, 30)
    )

    for (x, y, w, h) in face_locations:
        # Extrai a região da face detectada
        face = frame[y:y+h, x:x+w]

        # Salva a face temporariamente para calcular os embeddings
        temp_face_path = 'temp_face.jpg'
        cv2.imwrite(temp_face_path, face)

        # Obtenha o encoding da face detectada
        try:
            face_encoding = DeepFace.represent(img_path=temp_face_path, model_name='Facenet')[0]['embedding']
        except:
            continue

        # Compara a face detectada com as faces conhecidas
        name = "Unknown"
        max_similarity = 0.0
        for known_face_encoding, known_face_name in zip(known_face_encodings, known_face_names):
            similarity = DeepFace.verify(known_face_encoding, face_encoding, model_name='Facenet', distance_metric='cosine')['distance']
            if similarity < 0.5 and (1 - similarity) > max_similarity:  # Ajuste o valor conforme necessário
                max_similarity = 1 - similarity
                name = known_face_name

        if name == "Unknown":
            # Salva o rosto desconhecido no diretório específico com um nome aleatório
            random_name = str(uuid.uuid4())
            save_face(face, unknown_faces_dir, random_name)
        else:
            # Desenha um retângulo ao redor da face detectada
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # Desenha o nome abaixo da face detectada
            cv2.rectangle(frame, (x, y+h - 35), (x+w, y+h), (0, 255, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (x + 6, y+h - 6), font, 1.0, (255, 255, 255), 1)

    # Exibe o frame com as detecções
    cv2.imshow('Video', frame)

    # Sai do loop se a tecla 'q' for pressionada
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Libera os recursos
video_capture.release()
cv2.destroyAllWindows()