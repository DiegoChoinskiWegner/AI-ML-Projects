import cv2
import numpy as np

# Função para aplicar escala de cinza
def gray_scale(image):

    r, g, b = cv2.split(image)

    image_gray_scale = 0.114 * b + 0.587 * g + 0.299 * r
    image_gray_scale = image_gray_scale.astype("uint8")  # converte para tipo imagem (0-255)

    return image_gray_scale

# Função para aplicar escala binaria
def binary_scale_manual(image, limit):

    # Aplica binarização manual
    img_binary_color = np.where(image > limit, 255, 0).astype("uint8")

    return img_binary_color


# Função para aplicar escala binaria
def binary_scale(image, limit):

    _, img_binary_color = cv2.threshold(image, limit, 255, cv2.THRESH_BINARY)

    return img_binary_color


# Leitura da Imagem
imagemTeste = cv2.imread(r"C:\CAMINHO\DA\IMAGEM.jpg")

# Percentual de escala de tamanho(15%)
escala_percentual = 15

# Calcula o novo tamanho
largura = int(imagemTeste.shape[1] * escala_percentual / 100)
altura = int(imagemTeste.shape[0] * escala_percentual / 100)
dimensao = (largura, altura)

# Redimensiona a imagem
image_resized = cv2.resize(imagemTeste, dimensao)

cv2.imshow("Título", image_resized)

cv2.waitKey(0)                          # Espera uma tecla ser pressionada
cv2.destroyAllWindows()      


imagem_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)

img_gray = gray_scale(imagem_rgb)

cv2.imshow("Gray Scale", img_gray)

cv2.waitKey(0)                          # Espera uma tecla ser pressionada
cv2.destroyAllWindows()  


binary_img = binary_scale_manual(img_gray, 55)

cv2.imshow("Binary Manual", binary_img)


cv2.waitKey(0)                          # Espera uma tecla ser pressionada
cv2.destroyAllWindows()      

binary_img = binary_scale(img_gray, 65)

cv2.imshow("Binary", binary_img)


cv2.waitKey(0)                          # Espera uma tecla ser pressionada
cv2.destroyAllWindows()   