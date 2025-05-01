import cv2
import numpy as np
import requests
import time
import os
import csv
import torch
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from io import BytesIO
import pandas as pd
import threading
import matplotlib.pyplot as plt

losses = []
accuracies = []

class PokemonDataset(Dataset):
    """Dataset de Pokémon para entrenamiento"""
    def __init__(self, csv_file, img_dir, transform=None):
        self.pokemon_data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        # Crear un mapeo de nombres a índices para entrenamiento
        self.name_to_idx = {name: idx for idx, name in enumerate(self.pokemon_data['name'].unique())}
        self.idx_to_name = {idx: name for name, idx in self.name_to_idx.items()}
        
    def __len__(self):
        return len(self.pokemon_data)
    
    def __getitem__(self, idx):
        img_path = self.pokemon_data.iloc[idx]['image_path']
        pokemon_name = self.pokemon_data.iloc[idx]['name']
        
        # Verificar si la imagen existe
        if os.path.exists(img_path):
            image = Image.open(img_path).convert('RGB')
        else:
            # Imagen por defecto si no existe
            print(f"Advertencia: No se encontró la imagen {img_path}")
            image = Image.new('RGB', (224, 224), (255, 255, 255))
            
        if self.transform:
            image = self.transform(image)
            
        # Usar el mapeo para obtener el índice de clase
        label = self.name_to_idx[pokemon_name]
        return image, label

class PokemonRecognizer:
    def __init__(self):
        # Directorios para datos
        self.data_dir = "pokemon_data"
        self.img_dir = os.path.join(self.data_dir, "images")
        self.csv_path = os.path.join(self.data_dir, "pokemon_database.csv")
        self.model_path = os.path.join(self.data_dir, "pokemon_model.pth")
        self.class_mapping_path = os.path.join(self.data_dir, "class_mapping.pth")
        
        # Crear directorios si no existen
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.img_dir, exist_ok=True)
        
        # Definir transformaciones para las imágenes
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Cargar o crear base de datos de Pokémon
        self.pokedex = {}
        if os.path.exists(self.csv_path):
            print(f"Cargando base de datos de Pokémon desde {self.csv_path}...")
            self.load_from_csv()
        else:
            print("Base de datos no encontrada. Descargando desde PokeAPI...")
            self.download_pokemon_database()
        
        # Crear dataset
        self.dataset = PokemonDataset(
            csv_file=self.csv_path,
            img_dir=self.img_dir,
            transform=self.transform
        )
        
        # Cargar o inicializar modelo
        self.model = None
        self.initialize_model()
        
        # Mapeo de índices a nombres
        if os.path.exists(self.class_mapping_path):
            self.idx_to_name = torch.load(self.class_mapping_path)
        else:
            self.idx_to_name = self.dataset.idx_to_name
            torch.save(self.idx_to_name, self.class_mapping_path)
        
        # Variables para detección en tiempo real
        self.detection_active = False
        self.detection_thread = None
        self.current_prediction = None
        self.current_confidence = 0
        
        # Intervalo para predicciones (en segundos)
        self.prediction_interval = 0.5
        self.last_prediction_time = 0
    
    def load_from_csv(self):
        """Carga la base de datos de Pokémon desde un archivo CSV"""
        try:
            with open(self.csv_path, 'r', newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    name = row['name']
                    self.pokedex[name] = {
                        'id': int(row['id']),
                        'name': name,
                        'height': float(row['height']),
                        'weight': float(row['weight']),
                        'types': row['types'].split('|'),
                        'abilities': row['abilities'].split('|'),
                        'image_path': row['image_path'],
                        'description': row.get('description', 'No hay descripción disponible')  # Añadir descripción
                    }
            print(f"Base de datos cargada con {len(self.pokedex)} Pokémon")
        except Exception as e:
            print(f"Error al cargar desde CSV: {e}")
            print("Descargando datos nuevamente...")
            self.download_pokemon_database()
    
    def download_pokemon_database(self):
        """Descarga información de Pokémon desde PokeAPI y la guarda en CSV"""
        csv_data = [['id', 'name', 'height', 'weight', 'types', 'abilities', 'image_path', 'description']]
        
        for pokemon_id in range(1, 152):  # Hasta 1024 Pokémon
            try:
                print(f"Descargando información de Pokémon #{pokemon_id}...")
                response = requests.get(f"https://pokeapi.co/api/v2/pokemon/{pokemon_id}")
                if response.status_code == 200:
                    pokemon_data = response.json()
                    name = pokemon_data['name']
                    
                    # Procesar y guardar datos
                    types = [t['type']['name'] for t in pokemon_data['types']]
                    abilities = [a['ability']['name'] for a in pokemon_data['abilities']]
                    
                    # Guardar imagen
                    image_url = pokemon_data['sprites']['other']['official-artwork']['front_default']
                    image_path = os.path.join(self.img_dir, f"{pokemon_id}.png")
                    
                    # Descargar y guardar imagen
                    try:
                        img_response = requests.get(image_url)
                        if img_response.status_code == 200:
                            with open(image_path, 'wb') as img_file:
                                img_file.write(img_response.content)
                            print(f"Imagen guardada: {image_path}")
                        else:
                            image_path = "no_image"
                            print(f"No se pudo descargar imagen para {name}")
                    except Exception as img_error:
                        image_path = "no_image"
                        print(f"Error al guardar imagen para {name}: {img_error}")
                    
                    # Obtener descripción en español desde la API de especies
                    description = "No hay descripción disponible"
                    try:
                        species_url = pokemon_data['species']['url']
                        species_response = requests.get(species_url)
                        if species_response.status_code == 200:
                            species_data = species_response.json()
                            # Intentar encontrar una descripción en español
                            spanish_entries = [entry for entry in species_data['flavor_text_entries'] 
                                             if entry['language']['name'] == 'es']
                            if spanish_entries:
                                # Tomar la última descripción en español disponible
                                description = spanish_entries[-1]['flavor_text'].replace('\n', ' ').replace('\f', ' ')
                                print(f"Descripción en español obtenida para {name}")
                            else:
                                print(f"No se encontró descripción en español para {name}")
                    except Exception as desc_error:
                        print(f"Error al obtener descripción para {name}: {desc_error}")
                    
                    # Guardar en Pokedex
                    self.pokedex[name] = {
                        'id': pokemon_id,
                        'name': name,
                        'height': pokemon_data['height'] / 10,  # convertir a metros
                        'weight': pokemon_data['weight'] / 10,  # convertir a kg
                        'types': types,
                        'abilities': abilities,
                        'image_path': image_path,
                        'description': description
                    }
                    
                    # Añadir a datos CSV
                    csv_data.append([
                        pokemon_id,
                        name,
                        pokemon_data['height'] / 10,
                        pokemon_data['weight'] / 10,
                        '|'.join(types),
                        '|'.join(abilities),
                        image_path,
                        description
                    ])
                    
                    # Para no sobrecargar la API
                    time.sleep(0.5)
                    
            except Exception as e:
                print(f"Error al descargar Pokémon ID {pokemon_id}: {e}")
        
        # Guardar datos en CSV
        try:
            with open(self.csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(csv_data)
            print(f"Base de datos guardada en {self.csv_path}")
        except Exception as e:
            print(f"Error al guardar CSV: {e}")
    
    def initialize_model(self):
        """Carga un modelo existente o inicializa uno nuevo"""
        try:
            # Inicializar el modelo base
            num_classes = len(self.pokedex)
            self.model = models.resnet18(pretrained=True)
            self.model.fc = torch.nn.Linear(512, num_classes)
            
            # Verificar si existe un modelo guardado previamente
            if os.path.exists(self.model_path):
                print(f"Cargando modelo entrenado desde {self.model_path}...")
                self.model.load_state_dict(torch.load(self.model_path))
                self.model.eval()
                print("Modelo cargado exitosamente.")
            else:
                print("No se encontró un modelo entrenado. Se usará un modelo nuevo.")
                print("Es recomendable entrenar el modelo antes de usarlo (presiona 't' durante la ejecución).")
                self.model.eval()
        except Exception as e:
            print(f"Error al inicializar el modelo: {e}")
            print("Se inicializará un modelo básico...")
            self.model = models.resnet18(pretrained=True)
            self.model.fc = torch.nn.Linear(512, len(self.pokedex))
            self.model.eval()
    
    def train_model(self, epochs=5):
        """Entrena el modelo con las imágenes descargadas"""
        print("\n=== INICIANDO ENTRENAMIENTO DEL MODELO ===")
        
        # Crear conjunto de datos
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Dataset y DataLoader para entrenamiento
        train_dataset = PokemonDataset(
            csv_file=self.csv_path,
            img_dir=self.img_dir,
            transform=transform_train
        )
        
        if len(train_dataset) == 0:
            print("Error: El conjunto de datos está vacío")
            return False
        
        # Guardar el mapeo de índices a nombres
        self.idx_to_name = train_dataset.idx_to_name
        torch.save(self.idx_to_name, self.class_mapping_path)
        
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        
        # Preparar modelo para entrenamiento
        self.model.train()
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        print(f"Entrenando modelo por {epochs} épocas...")
        print(f"Total de imágenes para entrenamiento: {len(train_dataset)}")
        print(f"Total de clases (Pokémon): {len(train_dataset.name_to_idx)}")
        
        # Ciclo de entrenamiento
        for epoch in range(epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            
            print(f"\nÉpoca {epoch+1}/{epochs}")
            print("-" * 20)
            
            for i, (inputs, labels) in enumerate(train_loader):
                # Entrenar
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                # Estadísticas
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Mostrar progreso
                if (i+1) % 5 == 0:
                    print(f"Batch {i+1}/{len(train_loader)} | Loss: {running_loss/5:.4f} | Precisión: {100*correct/total:.2f}%")
                    running_loss = 0.0
            epoch_loss = running_loss / len(train_loader)
            epoch_accuracy = 100 * correct / total
            losses.append(epoch_loss)
            accuracies.append(epoch_accuracy)
            print(f"Época {epoch+1} completa | Loss: {epoch_loss:.4f} | Precisión: {epoch_accuracy:.2f}%")
        # Guardar modelo entrenado
        print("\nEntrenamiento completado. Guardando modelo...")
        torch.save(self.model.state_dict(), self.model_path)

        

# Visualización de pérdida
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(range(1, epochs+1), losses, label='Pérdida')
        plt.xlabel('Época')
        plt.ylabel('Loss')
        plt.title('Curva de Pérdida')
        plt.grid(True)
        plt.legend()

# Visualización de precisión
        plt.subplot(1, 2, 2)
        plt.plot(range(1, epochs+1), accuracies, label='Precisión', color='green')
        plt.xlabel('Época')
        plt.ylabel('Precisión (%)')
        plt.title('Curva de Precisión')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()
        
        # Volver a modo evaluación
        self.model.eval()
        print("Modelo guardado exitosamente en", self.model_path)
        return True
    
    def evaluate_model(self):
        """Evalúa el modelo con algunas imágenes de prueba"""
        print("\n=== EVALUANDO MODELO ===")
        
        # Crear dataloader para evaluación
        eval_dataset = PokemonDataset(
            csv_file=self.csv_path,
            img_dir=self.img_dir,
            transform=self.transform
        )
        
        eval_loader = DataLoader(eval_dataset, batch_size=16, shuffle=True)
        
        self.model.eval()
        correct = 0
        total = 0
        
        # No calcular gradientes durante la evaluación
        with torch.no_grad():
            for inputs, labels in eval_loader:
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f"Precisión del modelo: {accuracy:.2f}%")
    
    def predict_pokemon(self, image):
        """Identifica el Pokémon en la imagen usando el modelo entrenado"""
        # Verificar si el modelo existe
        if self.model is None:
            print("Error: El modelo no está inicializado")
            return None, 0
        
        # Convertir imagen de OpenCV a formato PIL
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # Preprocesar la imagen
        input_tensor = self.transform(pil_image)
        input_batch = input_tensor.unsqueeze(0)
        
        # Hacer predicción
        self.model.eval()
        with torch.no_grad():
            output = self.model(input_batch)
            probabilities = torch.nn.functional.softmax(output, dim=1)[0]
            prediction_idx = torch.argmax(output, 1).item()
            confidence = probabilities[prediction_idx].item() * 100
        
        # Obtener el nombre del Pokémon usando el mapeo
        try:
            pokemon_name = self.idx_to_name[prediction_idx]
            
            # Devolver la información del Pokémon
            if pokemon_name in self.pokedex:
                return self.pokedex[pokemon_name], confidence
            else:
                print(f"Error: Pokémon '{pokemon_name}' no encontrado en la base de datos")
                return None, 0
        except KeyError:
            print(f"Error: Índice {prediction_idx} no encontrado en el mapeo de clases")
            return None, 0
    
    def display_pokemon_info(self, frame, pokemon_info, confidence=None):
        """Muestra la información del Pokémon sobre el frame de video"""
        if not pokemon_info:
            cv2.putText(frame, "No se pudo identificar el Pokémon", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return frame
        
        # Crear una copia del frame para no modificar el original
        display_frame = frame.copy()
        
        # Información general en la parte superior
        name_text = f"{pokemon_info['name'].upper()} (#{pokemon_info['id']})"
        cv2.putText(display_frame, name_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        if confidence is not None:
            conf_text = f"Confianza: {confidence:.2f}%"
            cv2.putText(display_frame, conf_text, (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Crear un rectángulo semitransparente para el resto de la información
        h, w = display_frame.shape[:2]
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (0, h-220), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame)
        
        # Añadir información detallada
        y_pos = h - 190
        infos = [
            f"Tipos: {', '.join(pokemon_info['types'])}",
            f"Altura: {pokemon_info['height']} m",
            f"Peso: {pokemon_info['weight']} kg",
            f"Habilidades: {', '.join(pokemon_info['abilities'])}"
        ]
        
        for info in infos:
            cv2.putText(display_frame, info, (10, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y_pos += 30
        
        # Mostrar descripción (partida en líneas si es larga)
        description = pokemon_info.get('description', 'No hay descripción disponible')
        desc_words = description.split()
        desc_lines = []
        current_line = ""
        
        for word in desc_words:
            if len(current_line + " " + word) > 70:  # Limitar a ~70 caracteres por línea
                desc_lines.append(current_line)
                current_line = word
            else:
                if current_line:
                    current_line += " " + word
                else:
                    current_line = word
        
        if current_line:
            desc_lines.append(current_line)
        
        # Mostrar "Descripción:" como título
        cv2.putText(display_frame, "Descripcion:", (10, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_pos += 25
        
        # Mostrar cada línea de la descripción
        for line in desc_lines[:3]:  # Limitar a 3 líneas para no ocupar demasiado espacio
            cv2.putText(display_frame, line, (10, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_pos += 25
        
        # Si hay más líneas, añadir puntos suspensivos
        if len(desc_lines) > 3:
            cv2.putText(display_frame, "...", (10, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Intentar mostrar imagen del Pokémon en una esquina
        image_path = pokemon_info['image_path']
        if os.path.exists(image_path):
            pokemon_img = cv2.imread(image_path)
            if pokemon_img is not None:
                # Redimensionar la imagen para que no sea demasiado grande
                target_height = 100
                aspect_ratio = pokemon_img.shape[1] / pokemon_img.shape[0]
                target_width = int(target_height * aspect_ratio)
                pokemon_img = cv2.resize(pokemon_img, (target_width, target_height))
                
                # Colocar la imagen en la esquina superior derecha
                x_offset = w - target_width - 10
                y_offset = 10
                
                # Crear una máscara para fusionar la imagen del Pokémon (suponiendo que tiene fondo blanco)
                # Esto funciona mejor con imágenes PNG con transparencia
                try:
                    # Región de interés
                    roi = display_frame[y_offset:y_offset+target_height, x_offset:x_offset+target_width]
                    
                    # Crear una máscara del Pokémon (suponiendo fondo blanco)
                    img_gray = cv2.cvtColor(pokemon_img, cv2.COLOR_BGR2GRAY)
                    _, mask = cv2.threshold(img_gray, 240, 255, cv2.THRESH_BINARY_INV)
                    
                    # Invertir la máscara para el fondo
                    mask_inv = cv2.bitwise_not(mask)
                    
                    # Fondo negro para la región del Pokémon
                    img_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
                    
                    # Primer plano con la imagen del Pokémon
                    img_fg = cv2.bitwise_and(pokemon_img, pokemon_img, mask=mask)
                    
                    # Combinar ambas imágenes
                    dst = cv2.add(img_bg, img_fg)
                    
                    # Colocar la imagen combinada de vuelta en el frame
                    display_frame[y_offset:y_offset+target_height, x_offset:x_offset+target_width] = dst
                except Exception as e:
                    # Si la fusión falla, simplemente superponer la imagen
                    display_frame[y_offset:y_offset+target_height, x_offset:x_offset+target_width] = pokemon_img
                                
        return display_frame
    
    def continuous_detection(self, cap):
        """Ejecuta la detección continua de Pokémon en un hilo separado"""
        print("Deteccion en tiempo real activada")
        
        while self.detection_active and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Error: No se pudo obtener frame")
                self.detection_active = False
                break
            
            # Verificar si ha pasado suficiente tiempo desde la última predicción
            current_time = time.time()
            if current_time - self.last_prediction_time >= self.prediction_interval:
                # Realizar predicción
                pokemon_info, confidence = self.predict_pokemon(frame)
                
                # Actualizar variables de estado
                self.current_prediction = pokemon_info
                self.current_confidence = confidence
                self.last_prediction_time = current_time
        
        print("Detección en tiempo real desactivada")
    
    def run_camera(self):
        """Ejecuta la captura de cámara y el reconocimiento"""
        # Inicializar cámara
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: No se pudo abrir la cámara")
            return
        
        print("\nCámara iniciada.")
        print("Controles:")
        print("- 'c': Capturar y reconocer Pokémon")
        print("- 'r': Activar/desactivar reconocimiento en tiempo real")
        print("- 't': Entrenar modelo (5 épocas por defecto)")
        print("- 'e': Evaluar precisión del modelo")
        print("- 'q': Salir")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: No se pudo obtener frame")
                break
            
            # Mostrar información si está activa la detección en tiempo real
            if self.detection_active and self.current_prediction:
                display_frame = self.display_pokemon_info(
                    frame, self.current_prediction, self.current_confidence)
                cv2.imshow('Reconocedor de Pokémon', display_frame)
            else:
                # Mostrar el estado de la detección en tiempo real
                status_text = "Deteccion en tiempo real: " + ("ACTIVA" if self.detection_active else "INACTIVA")
                status_color = (0, 255, 0) if self.detection_active else (0, 0, 255)
                
                # Añadir texto al frame
                frame_with_text = frame.copy()
                cv2.putText(frame_with_text, status_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                
                # Instrucciones
                cv2.putText(frame_with_text, "Presiona 'r' para activar/desactivar", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                cv2.imshow('Reconocedor de Pokémon', frame_with_text)
            
            # Esperar a que el usuario presione una tecla
            key = cv2.waitKey(1) & 0xFF
            
            # Si presiona 'q', salir
            if key == ord('q'):
                # Detener la detección en tiempo real si está activa
                if self.detection_active:
                    self.detection_active = False
                    if self.detection_thread and self.detection_thread.is_alive():
                        self.detection_thread.join(timeout=1.0)
                break
            
            # Si presiona 'c', capturar y reconocer
            elif key == ord('c'):
                print("\nAnalizando imagen...")
                
                # Guardar la imagen capturada
                capture_path = os.path.join(self.data_dir, "captured.jpg")
                cv2.imwrite(capture_path, frame)
                print(f"Imagen guardada en {capture_path}")
                
                # Predecir Pokémon
                pokemon_info, confidence = self.predict_pokemon(frame)
                
                # Mostrar información en consola
                if pokemon_info:
                    print("\n" + "="*50)
                    print(f"POKÉMON IDENTIFICADO: {pokemon_info['name'].upper()} (#{pokemon_info['id']})")
                    if confidence is not None:
                        print(f"Confianza: {confidence:.2f}%")
                    print("="*50)
                    print(f"Tipos: {', '.join(pokemon_info['types'])}")
                    print(f"Altura: {pokemon_info['height']} m")
                    print(f"Peso: {pokemon_info['weight']} kg")
                    print(f"Habilidades: {', '.join(pokemon_info['abilities'])}")
                    print(f"Descripción: {pokemon_info.get('description', 'No disponible')}")
                    print("="*50 + "\n")
                else:
                    print("No se pudo identificar el Pokémon")
                
                # Mostrar información en la pantalla
                display_frame = self.display_pokemon_info(frame, pokemon_info, confidence)
                cv2.imshow('Pokémon Identificado', display_frame)
            
            # Si presiona 'r', activar/desactivar reconocimiento en tiempo real
            elif key == ord('r'):
                if self.detection_active:
                    # Desactivar la detección
                    self.detection_active = False
                    if self.detection_thread and self.detection_thread.is_alive():
                        self.detection_thread.join(timeout=1.0)
                    print("Detección en tiempo real desactivada")
                else:
                    # Activar la detección
                    self.detection_active = True
                    self.detection_thread = threading.Thread(
                        target=self.continuous_detection, args=(cap,))
                    self.detection_thread.daemon = True
                    self.detection_thread.start()
                    print("Detección en tiempo real activada")
            
            # Si presiona 't', entrenar modelo
            elif key == ord('t'):
                print("\nIniciando entrenamiento del modelo...")
                success = self.train_model(epochs=10)
                if success:
                    print("Entrenamiento completado con éxito.")
                    self.evaluate_model()
            
            # Si presiona 'e', evaluar modelo
            elif key == ord('e'):
                print("\nEvaluando modelo...")
                self.evaluate_model()
        
        # Liberar recursos
        cap.release()
        cv2.destroyAllWindows()


    
    # Verificar si hay un modelo
# Ejecutar el programa
if __name__ == "__main__":
    print("======== SISTEMA DE RECONOCIMIENTO DE POKÉMON ========")
    print("Inicializando...")
    recognizer = PokemonRecognizer()
    
    # Verificar si hay un modelo entrenado
    if not os.path.exists(recognizer.model_path):
        print("\nNo se encontró un modelo entrenado.")
        entrenar = input("¿Deseas entrenar el modelo ahora? (s/n): ").lower()
        if entrenar == 's':
            recognizer.train_model(epochs=15)
    
    recognizer.run_camera()