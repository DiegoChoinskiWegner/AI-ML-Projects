import tkinter as tk
from tkinter import messagebox, filedialog
import cv2
import numpy as np
import threading
import time
from PIL import Image, ImageTk

# --- Configurações dos Modelos ---
MODEL_DIR = './models/'
# Modelos para detecção de rostos (SSD)
PROTOTXT_PATH = MODEL_DIR + 'deploy.prototxt'
MODEL_PATH = MODEL_DIR + 'res10_300x300_ssd_iter_140000.caffemodel'
# Modelo para extração de embeddings (OpenFace)
EMBEDDING_MODEL_PATH = MODEL_DIR + 'nn4.small2.v1.t7'
CONFIDENCE_THRESHOLD = 0.5

# --- Classes para a Lógica de Reconhecimento ---

class FaceRecognizer:
    def __init__(self):
        # Carrega o detector de faces
        self.detector = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, MODEL_PATH)
        # Carrega o modelo de embeddings para reconhecimento
        self.embedder = cv2.dnn.readNetFromTorch(EMBEDDING_MODEL_PATH)
        self.known_faces = {}

    def add_known_face(self, name, image_path):
        """Adiciona uma pessoa ao banco de dados."""
        try:
            image = cv2.imread(image_path)
            (h, w) = image.shape[:2]
            
            # Detecção de rosto na imagem de referência
            blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
            self.detector.setInput(blob)
            detections = self.detector.forward()
            
            if detections.shape[2] > 0 and detections[0, 0, 0, 2] > CONFIDENCE_THRESHOLD:
                box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                face = image[startY:endY, startX:endX]
                face_blob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
                self.embedder.setInput(face_blob)
                embedding = self.embedder.forward().flatten()
                
                self.known_faces[name] = {'embedding': embedding, 'appearances': 0, 'screen_time': 0}
                return True
        except Exception as e:
            messagebox.showerror("Erro", f"Não foi possível adicionar a pessoa {name}: {e}")
            return False
        return False

    def recognize_face(self, frame):
        """Detecta e reconhece rostos em um quadro."""
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.detector.setInput(blob)
        detections = self.detector.forward()
        
        recognized_people = {}
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > CONFIDENCE_THRESHOLD:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                face_roi = frame[startY:endY, startX:endX]
                if face_roi.shape[0] < 20 or face_roi.shape[1] < 20:
                    continue

                face_blob = cv2.dnn.blobFromImage(face_roi, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
                self.embedder.setInput(face_blob)
                embedding = self.embedder.forward().flatten()
                
                name = "Desconhecido"
                min_dist = float('inf')
                
                for known_name, known_data in self.known_faces.items():
                    dist = np.linalg.norm(known_data['embedding'] - embedding)
                    if dist < min_dist:
                        min_dist = dist
                        name = known_name
                
                if min_dist > 0.8: # Limiar de similaridade (ajustável)
                    name = "Desconhecido"
                
                if name not in recognized_people:
                    recognized_people[name] = {'box': (startX, startY, endX, endY), 'embedding': embedding}

        return recognized_people

# --- Interface Gráfica Tkinter ---

class RecognitionApp:
    def __init__(self, master):
        self.master = master
        master.title("Reconhecimento de Faces em Vídeos")
        
        self.recognizer = FaceRecognizer()
        self.video_processing_thread = None
        self.stop_flag = threading.Event()
        self.video_path_var = tk.StringVar()

        self.video_source_var = tk.StringVar(value="None")
        self.setup_ui()

    def setup_ui(self):


        radio_button_frame = tk.Frame(self.master, padx=8, pady=8)
        radio_button_frame.pack(fill=tk.BOTH, expand=True)
        tk.Label(radio_button_frame, text="Escolha a origem do video:").pack(side=tk.LEFT)
        tk.Radiobutton(radio_button_frame , text="Arquivos locais", variable=self.video_source_var, value="local").pack(side=tk.LEFT)
        tk.Radiobutton(radio_button_frame , text="Videos Online", variable=self.video_source_var, value="online").pack(side=tk.LEFT)
        tk.Radiobutton(radio_button_frame , text="Camera", variable=self.video_source_var, value="camera").pack(side=tk.LEFT)


        # Frame contêiner input video
        self.input_container_frame = tk.Frame(self.master, padx=10, pady=10)
        self.input_container_frame.pack(fill=tk.X)

        # Adiciona rastreador à variável dos Radiobuttons
        self.video_source_var.trace_add('write', self.on_radio_change)  


        # Frame de gerenciamento de pessoas
        people_frame = tk.Frame(self.master, padx=10, pady=5)
        people_frame.pack(fill=tk.X)
        
        tk.Label(people_frame, text="Pessoas para Reconhecer:").pack(side=tk.LEFT)
        self.people_listbox = tk.Listbox(people_frame, selectmode=tk.MULTIPLE)
        self.people_listbox.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)

        tk.Button(people_frame, text="Adicionar Pessoa", command=self.add_person).pack(side=tk.LEFT, padx=5)
        
        # Frame de botões de controle
        control_frame = tk.Frame(self.master, padx=10, pady=10)
        control_frame.pack(fill=tk.X)
        tk.Button(control_frame, text="Iniciar Reconhecimento", command=self.start_recognition).pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="Finalizar Reconhecimento", command=self.stop_recognition).pack(side=tk.LEFT, padx=5)

        # Frame de controle de processamento
        process_control_frame = tk.Frame(self.master, padx=10, pady=10)
        process_control_frame.pack(fill=tk.BOTH, expand=True)
        tk.Label(process_control_frame, text="Procesamento:").pack(anchor=tk.W)
        self.process_control_text = tk.Text(process_control_frame, height=2, state=tk.DISABLED)
        self.process_control_text.pack(fill=tk.BOTH, expand=True)

        # Frame de resultados
        results_frame = tk.Frame(self.master, padx=10, pady=10)
        results_frame.pack(fill=tk.BOTH, expand=True)
        tk.Label(results_frame, text="Resultados:").pack(anchor=tk.W)
        self.results_text = tk.Text(results_frame, height=10, state=tk.DISABLED)
        self.results_text.pack(fill=tk.BOTH, expand=True)

        self.video_display_frame = tk.Frame(self.master, padx=10, pady=10)
        self.video_display_frame.pack(fill=tk.BOTH, expand=True)
        self.video_label = tk.Label(self.video_display_frame) # Onde o vídeo será mostrado
        self.video_label.pack()

    def on_radio_change(self, *args):
        """Função chamada quando um Radiobutton é selecionado."""
        # Limpa o conteúdo do frame contêiner
        for widget in self.input_container_frame.winfo_children():
            widget.destroy()

        selected_option = self.video_source_var.get()
        
        if selected_option == "local":
            tk.Label(self.input_container_frame, text="Caminho do Vídeo:").pack(side=tk.LEFT)
            self.video_path_entry = tk.Entry(self.input_container_frame, textvariable=self.video_path_var, width=50)
            self.video_path_entry.pack(side=tk.LEFT, padx=5, expand=True)
            tk.Button(self.input_container_frame, text="Navegar", command=self.browse_video).pack(side=tk.LEFT)
            self.video_path_var.set("")

        elif selected_option == "online":
            tk.Label(self.input_container_frame, text="URL do Vídeo:").pack(side=tk.LEFT)
            self.video_path_entry = tk.Entry(self.input_container_frame, textvariable=self.video_path_var, width=50)
            self.video_path_entry.pack(side=tk.LEFT, padx=5, expand=True)
            self.video_path_var.set("Insira o URL aqui...")
            
        elif selected_option == "camera":
            self.video_path_var.set("0")
            tk.Label(self.input_container_frame, text="Câmera selecionada.").pack(side=tk.LEFT)
            tk.Button(self.input_container_frame, text="Acessar Camera", command=self.capture_camera).pack(side=tk.LEFT)
            


    def browse_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi;*.mov;*.mkv")])
        if file_path:
            self.video_path_var.set(file_path)

    def start_video_feed(self, source):
        try:
            # Tenta converter a fonte para inteiro (para câmeras)
            video_source = int(source)
        except ValueError:
            # Se não for um número, usa como string (para arquivos/URLs)
            video_source = source

        self.video_capture = cv2.VideoCapture(video_source)
        if not self.video_capture.isOpened():
            messagebox.showerror("Erro", "Não foi possível abrir a fonte de vídeo.")
            return
            
        self.update_video_feed()

    def add_person(self):
        name = tk.simpledialog.askstring("Adicionar Pessoa", "Nome da Pessoa:")
        if name:
            image_path = filedialog.askopenfilename(title="Selecione uma imagem de rosto")
            if image_path:
                if self.recognizer.add_known_face(name, image_path):
                    self.people_listbox.insert(tk.END, name)
    
    def start_recognition(self):
        if hasattr(self, 'video_capture') and self.video_capture.isOpened():
            messagebox.showinfo("Aviso", "O reconhecimento já está em andamento.")
            return

        video_source = self.video_path_var.get()

        # A verificação de '0' é o que diferencia a câmera das outras fontes
        if video_source == "0":
            try:
                source = int(video_source)
            except ValueError:
                messagebox.showerror("Erro", "ID da câmera inválido.")
                return
        elif not video_source:
            messagebox.showerror("Erro", "Por favor, selecione uma fonte de vídeo.")
            return
        else:
            source = video_source

        # Zera os resultados anteriores
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.results_text.config(state=tk.DISABLED)

        self.stop_flag.clear()

        self.video_capture = cv2.VideoCapture(source)
        if not self.video_capture.isOpened():
            messagebox.showerror("Erro", f"Não foi possível abrir a fonte de vídeo '{video_source}'.")
            return

        self.update_video_feed()
        
    def stop_recognition(self):
        self.stop_flag.set()
        if hasattr(self, 'video_capture') and self.video_capture.isOpened():
            self.video_capture.release()
            self.video_label.config(image='') 
            messagebox.showinfo("Finalizado", "Reconhecimento finalizado.")
            self.show_results() 

    def show_results(self):
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        for name, data in self.recognizer.known_faces.items():
            screen_time_sec = data['screen_time']
            self.results_text.insert(tk.END, f"Pessoa: {name}\n")
            self.results_text.insert(tk.END, f"Tempo de Tela: {screen_time_sec:.2f} segundos\n")
            self.results_text.insert(tk.END, f"Aparições: {data['appearances']} vezes\n\n")
        self.results_text.config(state=tk.DISABLED)


    def process_video(self, video_source):
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            messagebox.showerror("Erro", "Não foi possível abrir a fonte de vídeo. Verifique o caminho ou URL.")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        
        while not self.stop_flag.is_set():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Lógica de reconhecimento
            recognized_people = self.recognizer.recognize_face(frame)
            
            # Atualiza contadores
            for name, data in self.recognizer.known_faces.items():
                if name in recognized_people:
                    # A pessoa apareceu neste frame, atualiza o tempo e a contagem
                    data['screen_time'] += 1/fps
                    if data['appearances'] == 0 or (data['appearances'] > 0 and (time.time() - data['last_seen_time']) > 1): # Lógica simplista para aparições
                        data['appearances'] += 1
                        data['last_seen_time'] = time.time()
                
        cap.release()

    def update_video_feed(self):
        """Captura um frame, realiza o reconhecimento de face e exibe na tela."""
        if self.stop_flag.is_set():
            return

        ret, frame = self.video_capture.read()
        if not ret:
            # Se o vídeo terminou ou a câmera foi desconectada
            self.stop_recognition()
            return
        
        # Inverte o frame se for da câmera (efeito espelho)
        if self.video_source_var.get() == 'camera':
            frame = cv2.flip(frame, 1)

        # Chama a função de reconhecimento de faces que você já criou
        recognized_people = self.recognizer.recognize_face(frame)
        
        # Itera sobre as pessoas reconhecidas para desenhar as caixas
        for name, data in recognized_people.items():
            (startX, startY, endX, endY) = data['box']
            
            # Define a cor e o texto do rótulo
            color = (0, 255, 0) if name != "Desconhecido" else (0, 0, 255)
            label = f"{name}"
            
            # Desenha a caixa delimitadora
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            
            # Prepara o fundo para o texto
            y = startY - 15 if startY - 15 > 15 else startY + 15
            
            # Desenha o texto do rótulo
            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

        # Converte o frame do OpenCV (BGR) para o formato do Pillow (RGB)
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Converte para um objeto de imagem do Tkinter
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        
        # Atualiza o Label do vídeo na interface
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        # Chama esta função novamente após um curto intervalo de tempo
        self.master.after(15, self.update_video_feed)
        
# --- Inicialização da Aplicação ---
if __name__ == "__main__":
    root = tk.Tk()
    app = RecognitionApp(root)
    root.mainloop()