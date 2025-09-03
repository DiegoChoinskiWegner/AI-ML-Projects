// main.go
// -----------------------------------------------------------------------------
// Este é o arquivo principal do nosso servidor em Go. Ele usa a biblioteca
// padrão 'net/http' para criar um servidor web que recebe webhooks da Twilio.
// -----------------------------------------------------------------------------

package main

import (
	"bufio" // Importado para ler o arquivo linha por linha
	"fmt"
	"io"
	"log"
	"math/rand" // Importado para gerar números aleatórios
	"net/http"
	"os"
	"path/filepath"
	"strings" // Importado para manipulação de strings
	"time"    // Importado para usar o tempo atual como semente

	"github.com/joho/godotenv" // Para carregar variáveis de ambiente do arquivo .env
)

// --- VARIÁVEIS GLOBAIS PARA CREDENCIAIS ---
// Elas serão carregadas da função main.
var twilioAccountSid string
var twilioAuthToken string

// --- DIRETÓRIO PARA SALVAR ÁUDIOS ---
const audioDir = "audios_recebidos"
const corpusFilePath = "utils/frases.txt" // Caminho para o arquivo de frases

// webhookHandler é a função que processa as requisições da Twilio.
func webhookHandler(w http.ResponseWriter, r *http.Request) {
	// A Twilio envia dados como um formulário, então precisamos parseá-lo.
	if err := r.ParseForm(); err != nil {
		log.Printf("Erro ao parsear o formulário: %v", err)
		http.Error(w, "Erro no request", http.StatusBadRequest)
		return
	}

	// Extrai os dados do corpo da requisição.
	from := r.FormValue("From")
	numMedia := r.FormValue("NumMedia")
	messageSid := r.FormValue("MessageSid")

	log.Printf("Webhook recebido de %s. MessageSid: %s", from, messageSid)

	var twimlResponse string

	// Verifica se a mensagem contém mídia.
	if numMedia != "0" {
		mediaURL := r.FormValue("MediaUrl0")
		mediaType := r.FormValue("MediaContentType0")

		log.Printf("Mídia recebida: URL=%s, Tipo=%s", mediaURL, mediaType)

		// Verifica se é um arquivo de áudio
		if len(mediaType) > 5 && mediaType[:5] == "audio" {
			err := downloadAndSaveAudio(mediaURL, messageSid)
			if err != nil {
				log.Printf("ERRO ao processar áudio: %v", err)
				twimlResponse = createTwimlMessage("Desculpe, tive um problema ao processar seu áudio.")
			} else {
				log.Printf("Áudio %s.ogg salvo com sucesso.", messageSid)
				twimlResponse = createTwimlMessage("Obrigado! Recebi seu áudio. Quando quiser a próxima frase, digite 'estou pronto' novamente.")
			}
		} else {
			twimlResponse = createTwimlMessage("Recebi sua mídia, mas no momento só consigo processar áudios.")
		}
	} else {
		// Se for uma mensagem de texto.
		messageText := r.FormValue("Body")
		log.Printf("Mensagem de texto recebida: \"%s\"", messageText)

		responseText := handleTextMessage(messageText)
		twimlResponse = createTwimlMessage(responseText)
	}

	// Envia a resposta TwiML de volta para a Twilio.
	w.Header().Set("Content-Type", "text/xml")
	fmt.Fprint(w, twimlResponse)
}

// handleTextMessage processa a mensagem de texto recebida e retorna uma resposta apropriada.
func handleTextMessage(incomingText string) string {
	lowerText := strings.ToLower(incomingText)

	if strings.Contains(lowerText, "olá") || strings.Contains(lowerText, "oi") || strings.Contains(lowerText, "começar") {
		return "Olá! Bem-vindo ao Clonador de Voz. Para iniciar, preciso que grave uma série de frases. Quando estiver pronto, digite 'estou pronto'."
	}

	if strings.Contains(lowerText, "estou pronto") {
		frases, err := lerFrasesDoArquivo(corpusFilePath)
		if err != nil {
			log.Printf("ERRO ao ler o arquivo de frases: %v", err)
			return "Desculpe, estou com um problema para encontrar as frases no momento. Tente novamente mais tarde."
		}
		if len(frases) == 0 {
			return "Parece que não há frases cadastradas. Por favor, verifique o arquivo de configuração."
		}

		indiceAleatorio := rand.Intn(len(frases))
		fraseSelecionada := frases[indiceAleatorio]
		return fmt.Sprintf("Ótimo! Por favor, grave a seguinte frase: \"%s\"", fraseSelecionada)
	}

	if strings.Contains(lowerText, "ajuda") {
		return "Eu sou um assistente para clonagem de voz. Eu te envio uma frase e você me responde com um áudio gravando essa frase. Digite 'status' para ver seu progresso."
	}

	if strings.Contains(lowerText, "status") {
		return "Status: Você ainda não iniciou a gravação. Vamos começar?"
	}

	return fmt.Sprintf("Recebi sua mensagem: \"%s\". Se precisar de ajuda, digite 'ajuda'.", incomingText)
}

// lerFrasesDoArquivo lê um arquivo de texto e retorna uma lista (slice) de frases.
func lerFrasesDoArquivo(caminho string) ([]string, error) {
	arquivo, err := os.Open(caminho)
	if err != nil {
		return nil, fmt.Errorf("não foi possível abrir o arquivo %s: %w", caminho, err)
	}
	defer arquivo.Close()

	var frases []string
	scanner := bufio.NewScanner(arquivo)
	for scanner.Scan() {
		linha := strings.TrimSpace(scanner.Text())
		if linha != "" { // Ignora linhas em branco
			frases = append(frases, linha)
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("erro ao ler o arquivo %s: %w", caminho, err)
	}

	return frases, nil
}

// downloadAndSaveAudio baixa o arquivo de áudio da URL da Twilio e salva localmente.
func downloadAndSaveAudio(url string, messageSid string) error {
	client := &http.Client{}
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return fmt.Errorf("erro ao criar request: %w", err)
	}

	req.SetBasicAuth(twilioAccountSid, twilioAuthToken)
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("erro ao fazer download: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("download falhou com status: %s", resp.Status)
	}

	filename := fmt.Sprintf("%s.ogg", messageSid)
	filepath := filepath.Join(audioDir, filename)
	file, err := os.Create(filepath)
	if err != nil {
		return fmt.Errorf("erro ao criar arquivo: %w", err)
	}
	defer file.Close()

	_, err = io.Copy(file, resp.Body)
	if err != nil {
		return fmt.Errorf("erro ao salvar arquivo: %w", err)
	}

	return nil
}

// createTwimlMessage gera a string XML de resposta para a Twilio.
func createTwimlMessage(message string) string {
	return fmt.Sprintf(`<?xml version="1.0" encoding="UTF-8"?><Response><Message>%s</Message></Response>`, message)
}

func main() {
	err := godotenv.Load()
	if err != nil {
		log.Println("Aviso: Não foi possível encontrar o arquivo .env. Usando variáveis de ambiente do sistema.")
	}

	twilioAccountSid = os.Getenv("TWILIO_ACCOUNT_SID")
	twilioAuthToken = os.Getenv("TWILIO_AUTH_TOKEN")

	if twilioAccountSid == "" || twilioAuthToken == "" {
		log.Fatal("As variáveis de ambiente TWILIO_ACCOUNT_SID e TWILIO_AUTH_TOKEN devem ser definidas.")
	}

	rand.Seed(time.Now().UnixNano())

	// Cria os diretórios necessários se eles não existirem.
	os.MkdirAll(filepath.Dir(corpusFilePath), 0755)
	os.MkdirAll(audioDir, 0755)

	http.HandleFunc("/webhook", webhookHandler)
	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, "Servidor do Clonador de Voz em Go está no ar!")
	})

	port := "3000"
	log.Printf("Servidor Go escutando na porta %s", port)
	log.Println("Para testar, use o ngrok para expor esta porta para a internet.")
	if err := http.ListenAndServe(":"+port, nil); err != nil {
		log.Fatalf("Erro ao iniciar o servidor: %v", err)
	}
}
