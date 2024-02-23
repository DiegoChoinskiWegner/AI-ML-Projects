Passos para criação do modelo de previsão e configurações.

1 - Criação do workspace ML na plataforma Azure:
	Dentro da plataforma Azure.microsoft após o login, devemos ir para a págia de recursos "Azure Machine Learning", dentro do recurso deve-se clica em "Criar" e prencher as informações necessárias, por padrão em 		laboratórios mantemos as opções que não estão vazias do mesmo jeito, em casos de uso reais devemos nos preocupar com dados que possam identificar de maneira facil o projeto, além de segurança e consistencia dos dados.
	Após a criação do ambiente de Machine Learninig clicamos na Workspace criada e logo em seguida em "iniciar o estúdio".
	A partir deste momente será aberta novas possibilidades de trabalho com Machine Learning e com modelos pré programados da plataforma, além da possibilidade de criar seus próprios modelos.

2 - Criação do modelo de com a base de dados escolhida:
	No Azure AI/ML Studio temos um menu com inúmeras ferramentas, durante este laboratório utilizamos a opção "ML automatizado" na parte de criação. Esta ferramenta nos dá a possibilidade de, segundo a documentação 		"automatizar as tarefas iterativas e demoradas do desenvolvimento de modelos de aprendizado de máquina. Ele permite que cientistas de dados, analistas e desenvolvedores criem modelos de ML com alta escala, eficiência 	e produtividade, ao mesmo tempo em que sustentam a qualidade do modelo." Dentro desta opão vamos criar uma nova tarefa de ML automatzada, é nesta etapa que vamos selecionar qual o tipo de tarefa que a IA deverá 		adotar e qual a base de dados será usada para o modelo(podendo esta ser documentos locais, links ou arquivos na nuvem).

3 - Criação da métricas de pontos finais:
	Após a criação do modelo podemos utiliza-lo através do botão "registrar modelo", em varias funcionalidades de visualização e teste, a partir dai que criamos os "Pontos Finais" ou EndPoins, esta etapa é um pouco mais 	complexa pois abre um leque maior de possibilidades, por isso optei por trazer o link da documentação "https://learn.microsoft.com/pt-pt/azure/machine-learning/how-to-deploy-online-endpoints?view=azureml-api-		2&tabs=azure-cli".

 Este processo durou com as opções padrão da Azure aproximadamente 35 a 45 minutos para estar prontamente configurado. Com opções de máquina, custo  e base de dados diferentes podemos ter outros niveis de performance. 
