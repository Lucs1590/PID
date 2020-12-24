# RECONHECIMENTO BIOMÉTRICO FACIAL E OCULAR
## Sumário

- MTCNN
- Descritores faciais
   - Local Binary Pattern (LBP)
   - VGGFace
- Métricas de desempenho
   - Precisão - Revocação
   - F-measure
   - AUC e ROC
   - CMC
   - Equal Error Rate (EER)
- Autenticação
- Referências


## Dataset

No ​ _dataset_ escolhido para a realização do trabalho foram utilizadas apenas 3257
imagens da face toda,
compreendendo 136 pessoas e 26 figuras diferentes de cada uma delas. Supõe-se
que o ​ _dataset_ ​ contou com novas imagens no decorrer dos anos.
A descrição das 26 categorias de imagens se dá por:

1. Expressão neutra;
2. Sorrindo;
3. Raiva;
4. Gritando;
5. Luz ao lado direito;
6. Luz ao lado esquerdo;
7. Todas as luzes acesas;
8. Usando óculos escuros;
9. Com óculos de sol e luz ao lado direito;
10.Com óculos de sol e luz ao lado esquerdo;
11.Usando cachecol;
12.Usando cachecol e luz à direita;
13.Com cachecol e luz à esquerda;
14 à 26. Segunda sessão (mesmas condições que 1 à 13).

## MTCNN

Para realizar a detecção facial, assim como os pontos do nariz, olho direito e
olho esquerdo e extremidades da boca, foram realizadas com o auxílio do
framework ​ _Multi-task Cascaded Convolutional Networks_ (MTCNN), baseada no
trabalho realizado por ZHANG. Kaipeng et al., 2016, sendo esta uma desenvolvida
como uma solução para a detecção e o alinhamento da face. Englobando três
estágios de redes convolucionais que são capazes de reconhecer rostos e pontos
de referência supracitados.
Nesse projeto, a sua implementação foi feita com o auxílio de uma biblioteca
já treinada, em Python, que recorre às tecnologias Tensorflow e Keras. A biblioteca
citada encontra-se disponível em: https://pypi.org/project/mtcnn/ <Acesso em nov.
2020>.

## Descritores faciais

### Local Binary Pattern (LBP)

Padrão binário local, ou em inglês, local binary pattern (LBP) é o nome que
se dá a técnica desenvolvida por OJALA, T., 2002, que consiste em computar e
representar localmente uma área pela sua textura e essa representação é devida ao
valor de um determinado pixel e seus vizinhos.
De modo sucinto, o processo do LBP consiste em transformar uma imagem
em escala de cinza e selecionar os pixels de uma determinada vizinhança a partir
do pixel central. Tendo feito isso, o pixel central passa a ser um limiar, em que os
vizinhos maiores ou iguais que o mesmo são transformados em 0 e os menores são
transformados em 1.
Tendo em vista que em um kernel 3x3 tem-se 8 vizinhos em relação ao pixel
central, pode-se dizer que se tem 2 ​^8 ​= 256 possibilidades de para padrões de
binários. Após o passo supracitado, um dos pixels vizinhos deve ser selecionado
para iniciar o cálculo do local, salvo que a posição desse pixel, assim como o
sentido (horário ou anti-horário), devem ser mantidos para todo o cálculo de LBP da
imagem. Após isso, os resultados dessa seleção binária são armazenados em um
vetor de 8 bits e posteriormente para decimal quando com o bit 1.
A saída desse processo resultará em uma imagem composta desses
números decimais e o último dos passos é computar um histograma de modo que
este contenha as características da imagem, ou melhor, os descritores faciais.

### VGGFace

O uso de ​ _deep learning_ tem crescido cada vez mais quando se trata de
identificação e trabalho com imagens, dessa forma dentre os modelos
desenvolvidos pela faculdade de Oxford que é considerado estado da arte para a
detecção e identificação facial, cita-se VGGFace e VGGFace2, as quais foram fruto
de outras redes do grupo Visual Geometry Group (VGG).
A denominada VGGFace foi publicada por Parkhi. 2015, em que o nome
original da publicação era ​ _Deep Face Recognition_ ​. Sua motivação era poder ser
comparada com redes como as do Google a Facebook, então realizou seu
treinamento com mais de dois milhões de faces, utilizando uma função de ativação
_softmax_ na camada de saída para classificar as faces como pessoas. Esta camada
é então removida para que a saída da rede seja uma representação vetorial da face.
Logo depois, em 2017, Qiong Cao, et al. escreveu um ​ _paper_ denominado
_VGGFace2: A dataset for recognizing faces across pose and age_ ​, em que sua
intenção era ter um ​ _dataset_ para a para treinar e avaliar modelos de reconhecimento
facial mais eficazes. O ​ _dataset_ continha 3,31 milhões de imagens de 9131
indivíduos, com uma média de 362,6 imagens para cada indivíduo. Os modelos são
treinados no conjunto de dados, especificamente um modelo ResNet-50 e um
modelo SqueezeNet-ResNet-50 (chamado SE-ResNet-50 ou SENet).
Quanto ao desenvolvimento do trabalho, foi utilizado da técnica de ​ _transfer
learning_ no modelo original da VGGFace2 com a resnet50, os quais estão
disponíveis em: https://github.com/rcmalli/keras-VGGFace <Acesso em dev. 2020>.
Para realizar a técnica de ​ _transfer learning_ ​ a última camada da rede foi retirada e
substituída por algumas camadas, sendo elas GlobalAveragePooling2D, em que o
_output_ de todas as celebridades utilizadas para essa rede (8631 celebridades)
passa a ser resumida em uma média, mantendo a profundidade.

## Métricas de desempenho

As métricas de desempenho, muito mais que mostrar os acertos e erros,
mostram o desempenho do modelo utilizado, tendo que esse pode ser avaliado por
diversos parâmetros.
Nessa seção alguns deles serão citados, de modo que seja possível,
também, ter um comparativo entre as redes desenvolvidas até o presente momento.

### Precisão - Revocação

A Precisão-Revocação é uma medida normalmente utilizada para previsão de
quando as classes estão muito desequilibradas. Na recuperação de informações, a
precisão é uma medida da relevância dos resultados, enquanto a revocação é uma
medida de quantos resultados verdadeiramente relevantes são devolvidos.
Em sua curva é possível visualizar a troca entre precisão e revocação para
diferentes limiares. Uma área alta sob a curva representa tanto a alta revocação
(baixa taxa de falsos negativos), quanto a alta precisão (baixa taxa de falsos
positivos). Pontuações altas, no geral, demonstra resultados precisos, bem como
sendo normalmente resultados positivos.
A precisão é definida como o número de verdadeiros positivos
sobre o número de verdadeiros positivos mais o número de falsos positivos.
Já a revocação é definida como o número de verdadeiros positivos
sobre o número de verdadeiros positivos mais o número de falsos negativos.

### AUC e ROC

Do inglês, Area Under the Curve (AUC) ou Area Under the Receiver
Operating Characteristic (AUROC) é uma métrica muito utilizada para problemas de
classificação que conta com vários ajustes de limiares. ROC é uma curva de
probabilidade e a AUC representa o grau ou medida de separabilidade, sendo que
essa métrica informa o quanto o modelo é capaz de distinguir entre as classes.
Quanto maior a AUC, melhor o modelo está prevendo as classes corretamente.
Quanto mais bem separada forem os falsos e positivos , melhor será a curva ROC.
A curva ROC é traçada com TPR (True Positive Receiver) contra a FPR
(False Positive Receiver) onde TPR está no eixo y, e FPR está no eixo x. Todavia,
vale destacar que em algumas documentações esses parâmetros podem variar.
Outro ponto a se destacar é que a curva ROC conta com um limiar que varia
entre o TPR e o FPR, sendo que esta variação pode ser de acordo com o problema
a ser solucionado. Um exemplo disso são os sistemas de segurança, que contam
com um FAR baixo e limiar alto. Em contrapartida, em sistemas forense o FAR pode
ser alto, tendo um limiar mais baixo.
Para exemplificar o comportamento de ROC e AUC, sendo ROC é uma curva
de probabilidade, serão traçadas distribuições de probabilidades, tendo a curva de
distribuição vermelha como classe positiva e a curva de distribuição verde como
classe negativa.
Uma situação ideal ocorre quando duas curvas não se sobrepõem, ou seja, o
modelo tem uma medida ideal de separação, sendo perfeitamente capaz de
distinguir as classes.

### CMC

Uma curva CMC é usada para avaliar a precisão dos algoritmos que
produzem uma lista ordenada de possíveis combinações entre valor provável e
porcentagem. No caso do reconhecimento facial, a saída do algoritmo (VGGFace,
por exemplo) seria uma lista de rostos, sendo que o rosto genuíno carregaria
consigo a maior probabilidade.
Em geral, quanto melhor for o algoritmo, maior será a porcentagem CMC de
classificação.

### Equal Error Rate (EER)

Equal Error Rate (EER) é uma métrica que normalmente é utilizada em
sistemas de segurança biométrica, de modo que predetermine os valores limiares
para sua taxa de falsa aceitação (FAR) e sua taxa de falsa rejeição (FRR). Quando
as taxas são iguais, o valor comum é referido como a ERR. Basicamente, seu valor
indica qual proporção de FAR é igual à proporção de FRR. Quanto menor o valor da
ERR, maior é a precisão do sistema biométrico.

O Equal Error Rate também pode ser chamado de taxa de erro cruzado
(​crossover rate​) ou taxa de erro cruzado, do inglês Crossover Error Rate (CER).


## Autenticação

Por fim, assim como descrito no rascunho da atividade, deveria ser
desenvolvido um modelo que dado duas imagens, fosse possível indicar a
autenticidade entre as mesmas. Para isso seria necessário utilizar de uma imagem
base para ser considerada autêntica e que fizesse uma comparação com as outras
imagens do ​ _dataset_ de modo que o modelo aprendesse as distâncias/diferenças
entre as mesmas, diminuindo essa distância entre os valores autênticos e
aumentasse entre os valores negativos. Algo que fica muito claro na visualização da
seguinte imagem que representa a ​ _Triple Loss_ ​, muito utilizada para esse
tipo de problema.
Além das redes siamesas que também foram propostas para esse trabalho,
para utilizar problemas ​ _multiclass_ ​, como é o caso deste problema, juntamente com
os modelo de SVM que foram utilizados juntamente com o LBP, costuma-se aplicar
o método ​ _One versus all_ ​, de modo que tenha-se um neurônio especialista para cada
uma das classes o qual “diria” se aquele input é autêntico ou não.
Porém, dentre as possíveis alternativas para a resolução desse problema, a
utilizada neste projeto contou com o auxílio da biblioteca scipy, em que com o
método cosine é possível realizar a verificação calculando a distância cosseno entre
a incorporação da identidade conhecida e as incorporações dos rostos dos
candidatos.
O alcance dessa distância varia entre 0 e 1, tendo um valor limiar entre as
identidades normalmente entre 0.4 e 0.6, sendo que nesse projeto o valor limiar
padrão foi de 0.5.
Para encontrar qual o valor das diferenças de distância cosseno, as amostras
passaram por um pré tratamento, seguido assim de uma predição por um dos
modelos que foram desenvolvidos no decorrer deste trabalho e por fim, uma
comparação.

## Referências

ZHANG, Kaipeng et al. Joint face detection and alignment using multitask cascaded
convolutional networks. IEEE Signal Processing Letters, v. 23, n. 10, p. 1499-1503,
2016.

OJALA, Timo; PIETIKAINEN, Matti; MAENPAA, Topi. Multiresolution gray-scale and
rotation invariant texture classification with local binary patterns. IEEE Transactions
on pattern analysis and machine intelligence, v. 24, n. 7, p. 971-987, 2002.


PARKHI, Omkar M.; VEDALDI, Andrea; ZISSERMAN, Andrew. Deep face
recognition. 2015.

CAO, Qiong et al. VGGFace2: A dataset for recognising faces across pose and age.
In: 2018 13th IEEE International Conference on Automatic Face & Gesture
Recognition (FG 2018). IEEE, 2018. p. 67-74.