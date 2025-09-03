import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
import speech_recognition as sr
import threading
import pyttsx3
from datetime import datetime
import numpy as np

# --- Configurações dos Modelos ---
MODEL_DIR = './models/'
PROTOTXT_PATH = MODEL_DIR + 'deploy.prototxt'
MODEL_PATH = MODEL_DIR + 'res10_300x300_ssd_iter_140000.caffemodel'
EMBEDDING_MODEL_PATH = MODEL_DIR + 'nn4.small2.v1.t7'
CONFIDENCE_THRESHOLD = 0.5

# --- Classes para a Lógica de Reconhecimento ---

class FaceRecognizer:
    def __init__(self):
        self.detector = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, MODEL_PATH)
        self.embedder = cv2.dnn.readNetFromTorch(EMBEDDING_MODEL_PATH)
        self.known_faces = {}

    def add_known_face(self, name, image_path):
        """Adiciona uma pessoa ao banco de dados a partir de um arquivo de imagem."""
        try:
            image = cv2.imread(image_path)
            if image is None:
                messagebox.showerror("Erro", f"Não foi possível ler a imagem em: {image_path}")
                return False
            
            (h, w) = image.shape[:2]
            
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
                
                self.known_faces[name.lower()] = {'embedding': embedding}
                print(f"Usuário '{name}' adicionado ao banco de dados facial.")
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
                if face_roi.shape[0] < 20 or face_roi.shape[1] < 20: continue

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
                
                if min_dist > 0.8: name = "Desconhecido"
                
                recognized_people[name] = {'box': (startX, startY, endX, endY)}
        return recognized_people
    

# Classe principal da nossa aplicação
class AssistenteGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Assistente Virtual - Autenticação")

        # --- Variáveis de Estado da Aplicação ---
        self.face_recognizer = FaceRecognizer()
        self.speech_recognizer = sr.Recognizer()
        
        # Máquina de estados para o novo fluxo
        self.login_step = "awaiting_hidden_voice"
        self.authenticated_user = None

        # --- Carregar Usuário Autorizado ---
        self.USER_NAME = "Diego"
        self.USER_IMAGE_PATH = r"C:\Users\bruna\Documents\Documentos pessoais\Foto concurso.jpeg"
        self.face_recognizer.add_known_face(self.USER_NAME, self.USER_IMAGE_PATH)
        
        # --- Configurações de TTS e Câmera ---
        self.tts_engine = pyttsx3.init()
        voices = self.tts_engine.getProperty('voices')
        self.tts_engine.setProperty('voice', voices[0].id)
        self.video_capture = cv2.VideoCapture(0)
        if not self.video_capture.isOpened():
            messagebox.showerror("Erro", "Não foi possível abrir a câmera.")
            return

        # --- Configuração do Reconhecimento de Voz ---
        self.microphone = sr.Microphone()
        with self.microphone as source:
            self.speech_recognizer.adjust_for_ambient_noise(source, duration=1)
        self.stop_listening = self.speech_recognizer.listen_in_background(self.microphone, self._callback_audio)
        print("Assistente pronto. Ouvindo em segundo plano...")

        # --- Estrutura da Interface (Tkinter) ---
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        login_frame = tk.Frame(main_frame, padx=10, pady=10)
        login_frame.grid(row=0, column=0, sticky="ew")
        
        # O botão de login começa desabilitado
        self.login_button = ttk.Button(login_frame, text="Iniciar Login (Rosto -> Íris)", command=self.start_login_process, state=tk.DISABLED)
        self.login_button.pack(side=tk.LEFT)

        self.video_label = ttk.Label(main_frame)
        self.video_label.grid(row=1, column=0, pady=10)
        
        self.status_label = ttk.Label(main_frame, text="Aguardando...")
        self.status_label.grid(row=2, column=0, pady=5, sticky="ew")
        
        # --- Iniciar os processos ---
        self.update_video_feed()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Inicia a primeira etapa "oculta" da autenticação
        self._falar_em_thread("Hello, What's your name?")

    def start_login_process(self):
        """Inicia a sequência visível de autenticação (Rosto -> Íris)."""
        if self.login_step == "voice_authenticated_idle":
            self.login_step = "awaiting_face"
            self.status_label.config(text="Olhe para a câmera para reconhecimento facial.")
            self._falar_em_thread("Please, look at the camera for facial recognition.")
        else:
            print("A etapa de voz ainda não foi concluída.")

    def _callback_audio(self, recognizer, audio):
        """Processa a autenticação por voz 'oculta'."""
        if self.login_step != "awaiting_hidden_voice":
            return

        try:
            texto = recognizer.recognize_google(audio, language="pt-BR").lower()
            print(f"Áudio detectado: '{texto}'")
            
            if "barry" in texto:
                # Voz correta! Muda o estado e habilita o botão de login.
                self.login_step = "voice_authenticated_idle"
                self.status_label.config(text=f"Olá, {self.USER_NAME}. Pressione o botão para continuar.")
                self.login_button.config(state=tk.NORMAL) # HABILITA O BOTÃO
                self._falar_em_thread(f"Hello, {self.USER_NAME}.")
        
        except (sr.UnknownValueError, sr.RequestError):
            pass

    def _ativar_assistente(self):
        """Lida com a lógica de ativação final."""
        hora_atual = datetime.now().hour
        if 5 <= hora_atual < 12: saudacao = f"Bom dia, {self.authenticated_user}"
        elif 12 <= hora_atual < 18: saudacao = f"Boa tarde, {self.authenticated_user}"
        else: saudacao = f"Boa noite, {self.authenticated_user}"

        texto_resposta = f"{saudacao}. Acesso liberado."
        self.status_label.config(text=texto_resposta)
        self._falar_em_thread(texto_resposta)

    def _falar_em_thread(self, texto):
        threading.Thread(target=self._falar, args=(texto,)).start()

    def _falar(self, texto):
        self.tts_engine.say(texto)
        self.tts_engine.runAndWait()

    def update_video_feed(self):
        """Exibe o vídeo e executa o reconhecimento facial quando solicitado."""
        ret, frame = self.video_capture.read()
        if not ret:
            self.root.after(15, self.update_video_feed)
            return

        frame = cv2.flip(frame, 1)
        
        # Lógica de reconhecimento facial só é executada na etapa correta
        if self.login_step == "awaiting_face":
            recognized_people = self.face_recognizer.recognize_face(frame)
            
            for name, data in recognized_people.items():
                (startX, startY, endX, endY) = data['box']
                color = (0, 255, 0) if name.lower() == self.USER_NAME.lower() else (0, 0, 255)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                cv2.putText(frame, name, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                if name.lower() == self.USER_NAME.lower():
                    # Rosto correto! Passa para a próxima etapa (Íris)
                    self.login_step = "awaiting_iris" # Mude para "authenticated" se não houver íris
                    self.status_label.config(text="Reconhecimento facial completo. Preparando leitura de íris...")
                    self._falar_em_thread("Facial recognition complete. Preparing for iris scan.")
                    
                    # --- PONTO PARA ADICIONAR A LÓGICA DA ÍRIS ---
                    # Como ainda não temos a íris, vamos autenticar diretamente
                    self.authenticated_user = self.USER_NAME
                    self.login_step = "authenticated"
                    self._ativar_assistente()
                    
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        self.root.after(15, self.update_video_feed)

    def on_closing(self):
        print("Fechando a aplicação...")
        self.login_step = "idle"
        if self.stop_listening:
            self.stop_listening(wait_for_stop=False)
        self.video_capture.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = AssistenteGUI(root)
    root.mainloop()
