# -- coding: utf-8 --
"""
Created on Wed Dec 4 14:33:56 2024

@author: defariaa
"""

from deepface import DeepFace
import sys
import os
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from mtcnn import MTCNN
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
from tensorflow.keras.utils import to_categorical
import ast
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import (
	LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
) 
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score



models = [
  "VGG-Face", 
  "Facenet", 
  "Facenet512", 
  "OpenFace", 
  "DeepFace", 
  "DeepID", 
  "ArcFace", 
  "Dlib", 
  "SFace",
  "GhostFaceNet",
]

backends = [
  'opencv', 
  'ssd', 
  'dlib', 
  'mtcnn', 
  'fastmtcnn',
  'retinaface', 
  'mediapipe',
  'yolov8',
  'yunet',
  'centerface',
]

alignment_modes = [True, False]


CLASSIFIER = "MLP"   # SVM, GNB
MODEL = "Facenet512" # VGG-Face Facenet512
BACKEND = "mtcnn"

ROOT_DIR = "./Base_photo_2024_v2"
CHECKPOINT_DATA = "./save_embeddings_photos"
DATA_AUMENGTE = "./Base_photo_2024_v2_augmente"

db_auths = {}

def adjust_luminosity(base_dir, increase_factor=1.5, decrease_factor=0.5):
    def change_luminosity(img, factor):
        """Change la luminosité d'une image en multipliant par un facteur."""
        if img is None:
            return None
        return np.clip(img * factor, 0, 255).astype(np.uint8)

    for person in os.listdir(base_dir):
        person_dir = os.path.join(base_dir, person)
        
        # Vérifier si c'est un dossier
        if not os.path.isdir(person_dir):
            print(f"{person_dir} n'est pas un dossier. Ignoré.")
            continue

        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            
            # Vérifier si c'est bien un fichier image valide
            if not os.path.isfile(img_path) or not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                print(f"{img_path} n'est pas une image valide. Ignoré.")
                continue
            
            img = cv2.imread(img_path)
            if img is None:
                print(f"Impossible de lire l'image : {img_path}. Fichier ignoré.")
                continue
            
            try:
                # Augmenter et diminuer la luminosité
                img_brighter = change_luminosity(img, increase_factor)
                img_darker = change_luminosity(img, decrease_factor)
                
                # Sauvegarder les images avec nouvelle luminosité
                if img_brighter is not None:
                    bright_path = os.path.join(person_dir, f"bright_{img_name}")
                    cv2.imwrite(bright_path, img_brighter)
                
                if img_darker is not None:
                    dark_path = os.path.join(person_dir, f"dark_{img_name}")
                    cv2.imwrite(dark_path, img_darker)
            
            except Exception as e:
                print(f"Erreur lors du traitement de {img_path} : {e}")


#adjust_luminosity(DATA_AUMENGTE)
#sys.exit()



#%%
def generate_embeddings(model_name):
    embeddings = []
    target = []
    images_paths = []
    
    embeddings_csv = CHECKPOINT_DATA + model_name + ".csv"
    print(embeddings_csv)
    
    if os.path.isfile(embeddings_csv):
        data = pd.read_csv(embeddings_csv, index_col=0)

    else:
        print("Entra no else")
        for author in os.listdir(DATA_AUMENGTE):
            db_auths[author] = {"img_path": [], "embedding": []}
            
            for image_path in glob(os.path.join(DATA_AUMENGTE, author, "*")):
                
                if not image_path.endswith(".jpg") and not image_path.endswith(".jpeg"):
                    continue
                
                print(image_path)
                
                db_auths[author]["img_path"].append(image_path)
                
                img = cv2.imread(image_path, cv2.IMREAD_COLOR)

                results = DeepFace.represent(
                    img_path = img, 
                    model_name = model_name,
                    detector_backend = "skip",
                    align = alignment_modes[1],
                    enforce_detection=False
                    )

                embeddings.append(results[0]["embedding"])
                target.append(author)
                images_paths.append(image_path)
                
                db_auths[author]["embedding"].append(results[0]["embedding"])

        print(embeddings)
        data = pd.DataFrame(embeddings, columns=[f'feature_{i}' for i in range(len(embeddings[0]))])

        data["author"] = target 
        data["images_paths"] = images_paths
        data.to_csv(embeddings_csv)
    return data


def train_evaluate_classifiers(data_pd):
    Y = data_pd["author"]
    X = data_pd.drop(["author"], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
		X, Y, test_size=0.2, stratify=Y, 
		random_state=42
	)

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)


    mlp = MLPClassifier(
		hidden_layer_sizes=(100, 50), 
		activation='relu', 
		solver='adam', 
		max_iter=500,
		random_state=7
	)
    svm = SVC(kernel='rbf', probability=True, random_state=1)
    knn = KNeighborsClassifier(n_neighbors=5)
    lda = LinearDiscriminantAnalysis() 
    qda = QuadraticDiscriminantAnalysis()
    gnb = GaussianNB()
    models = [mlp, svm, knn, lda, qda, gnb]
    name_model = ["MLP", "SVM", "KNN", "LDA", "QDA", "GNB"]
    accuracy = []
    acc_confiance = []

    for i, classifier in enumerate(models):
        classifier.fit(X_train.drop("images_paths", axis=1), y_train_encoded)

		# Évaluer le modèle
        y_pred = classifier.predict(X_test.drop("images_paths", axis=1))
        print(f"Rapport de classification {name_model[i]} :")
        print(classification_report(y_test_encoded, y_pred, target_names=label_encoder.classes_))

        disp = ConfusionMatrixDisplay(
			confusion_matrix(y_test_encoded, y_pred), 
			display_labels=label_encoder.classes_
		)
        disp.plot()
        plt.title(f"Matrice de confusion - VGG16 et {name_model[i]}")
        plt.show()
        
        acc = accuracy_score(y_test_encoded, y_pred)
        # Calculer l'intervalle de confiance
        n = len(y_test_encoded)
        acc_conf = 1.96 * np.sqrt((acc * (1 - acc)) / n)
        accuracy.append(acc*100)
        acc_confiance.append(acc_conf*100)

    return models, name_model, accuracy, acc_confiance, label_encoder.classes_


def train_model_identification():
	
    data = generate_embeddings("VGG-Face")
    data_face = generate_embeddings("Facenet512")
    
    models_vgg, name_model, accuracy_vgg, acc_conf_vgg, name_classes = train_evaluate_classifiers(data)
    models_face, name_model, accuracy_facenet, acc_conf_facenet, name_classes = train_evaluate_classifiers(data_face)
        

    # Dados de exemplo
    categorias = name_model
    valores_azul = accuracy_facenet
    valores_vermelho = accuracy_vgg
    
    # Definindo a posição das barras
    x = np.arange(len(categorias))  # posição das categorias
    largura = 0.35  # Largura das barras
    
    # Criando o gráfico de barras
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Barras azuis e vermelhas
    bars_azul = ax.bar(x - largura/2, valores_azul, largura, label='Facenet512', yerr=acc_conf_facenet, color='blue')
    bars_vermelho = ax.bar(x + largura/2, valores_vermelho, largura, label='VGG-Face', yerr=acc_conf_vgg, color='red')
    
    # Adicionando rótulos e título
    ax.set_xlabel('Classificateurs')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracies Classificateurs avec VFGG_Face et Facenet512')
    ax.set_xticks(x)
    ax.set_xticklabels(categorias)
    ax.legend()
    
    # Adicionando os valores em cima das barras
    for bar in bars_azul:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f}%', ha='center', va='bottom', fontsize=10)
    
    for bar in bars_vermelho:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f}%', ha='center', va='bottom', fontsize=10)
    
    
    ax.legend(loc='lower right')
    
    # Exibindo o gráfico
    plt.tight_layout()
    plt.show()
    
    if MODEL == "Facenet512":
        classifier = models_face[name_model.index(CLASSIFIER)]
    elif MODEL == "VGG-Face":
        classifier = models_vgg[name_model.index(CLASSIFIER)]
        
    return classifier, name_classes
	


def classify_real_time():

	#classifier = SVC(kernel='rbf', probability=True, random_state=1)
	#classifier = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=500)

	# Entraîner le classifieur SVM
	classifier, classes_names = train_model_identification()



	# Démarrer la capture vidéo
	cap = cv2.VideoCapture(0)

	while cap.isOpened():
		ret, frame = cap.read()
		if not ret:
			print("Erreur : Impossible de capturer la vidéo.")
			break

		# Convertir le cadre en RGB pour le détecteur de visages
		frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

		# Détecter les visages dans le cadre
		detections = DeepFace.extract_faces(
			img_path=frame_rgb,
			detector_backend=BACKEND,
		    align = alignment_modes[0],
			enforce_detection=False
		)

		for result in detections:
			bounding_box = result['facial_area']
			x, y, w, h, _, _ = list(bounding_box.values())

			print("x =", x)
			print("y =", y)
			print("w =", w)
			print("h =", h)


			# Extraire la région du visage
			face = frame_rgb[max(0, int(y)):max(0, int(y) + int(h)), max(0, int(x)):max(0, int(x) + int(w))]
			if face.size == 0:
				print("Erreur : Impossible d'extraire le visage.")
				continue

			
			#plt.imshow(face)  # Exibe a imagem
			#plt.axis('off')  # Desliga os eixos
			#plt.title("Image coupé")
			#plt.show()  # Exibe a janela do Matplotlib

			try:
				# Redimensionner et prétraiter l'image du visage
				embedding_obj = DeepFace.represent(
					img_path = face,
					detector_backend = "skip",
					model_name = MODEL,
					align = alignment_modes[1],
				)[0]["embedding"]

				print("embedding_obj.shape =", len(embedding_obj))

				embedding_array = np.array(embedding_obj).reshape(1, -1)
				
				face_resized = cv2.resize(face, (224, 224))
				img_array = image.img_to_array(face_resized)
				img_array = np.expand_dims(img_array, axis=0)
				img_array = preprocess_input(img_array)

				print("embedding_array.shape =", embedding_array.shape)

                # Prédire l'identité à l'aide du classifieur
				id_author = classifier.predict(embedding_array)
				author = classes_names[id_author]
				confidence = np.max(classifier.predict_proba(embedding_array))

                # Dessiner la boîte englobante et afficher le nom et la confiance
				cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
				cv2.putText(frame, f"{author} ({confidence:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

			except Exception as e:
				print(f"Erreur lors du traitement d'un visage : {e}")

        # Afficher le cadre avec les résultats
		cv2.imshow("Real-Time Classification", frame)

        # Quitter avec la touche 'q'
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

    # Libérer les ressources
	cap.release()
	cv2.destroyAllWindows()


classify_real_time()