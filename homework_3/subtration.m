clear all
# T_3
img1 = imread('Figuras2/Image2.bmp');
img2 = imread('Figuras2/Image3.bmp');
img3 = imabsdiff(img2,img1);
img4 = imcomplement(img3);
figure('Name', 'Imagem Original'),imshow(img1)
figure('Name', 'Histograma da Imagem Original'), imhist(img1)
figure('Name','Imagem do Fundo'), imshow(img2)
figure('Name','Histograma da Imagem do Fundo'), imhist(img2)
figure('Name', 'Imagem Diferença'), imshow(img3)
figure('Name', 'Histograma da Imagem Diferença'), imhist(img3)
figure('Name','Imagem resultado da subtração complementada'), imshow(img4)
figure('Name','Histograma da Imagem complementada'), imhist(img4)

# E_6
img5 = img2 - img1;
figure('Name', 'Subtração B e A Comum'),imshow(img5)
img6 = img1 - img2;
figure('Name', 'Subtração A e B Comum'),imshow(img6)

# E_7