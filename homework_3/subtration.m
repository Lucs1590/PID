clear all
# T_3
img1 = imread('images/Image2.bmp');
img2 = imread('images/Image3.bmp');
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
img_polem = imread('images/polem.bmp');
figure('Name', 'Imagem Original')
imshow(img_polem)
img_polem_clara = img_polem + 128;
figure('Name', 'Imagem clareada')
imshow(img_polem_clara)

img_p_eq = histeq(img_polem,128);
img_pc_eq = histeq(img_polem_clara,128);
figure('Name', 'Histograma da Imagem Original')
imhist(img_p_eq)
figure('Name', 'Histograma da Imagem Clara')
imhist(img_pc_eq)

img_polem = imread('images/polem.bmp');
figure('Name', 'Imagem Original')
imshow(img_polem)
img_polem_clara = img_polem + 200;
figure('Name', 'Imagem clareada')
imshow(img_polem_clara)

img_p_eq = histeq(img_polem,128);
img_pc_eq = histeq(img_polem_clara,128);
figure('Name', 'Histograma da Imagem Original')
imhist(img_p_eq)
figure('Name', 'Histograma da Imagem Clara')
imhist(img_pc_eq)

figure('Name', 'Imagem Original Equalizada')
imshow(img_p_eq)
figure('Name', 'Imagem Clara Equalizada')
imshow(img_p_eq)

# E_8
raw_img = imread('images/ferramentas.bmp');
img_bin_1 = im2bw(raw_img,0.2);
img_bin_2 = im2bw(raw_img,0.3);
img_bin_3 = im2bw(raw_img,0.4);
img_bin_4 = im2bw(raw_img,0.5);
img_bin_5 = im2bw(raw_img,0.6);
figure('Name', 'Imagem Bin 20%')
imshow(img_bin_1)
figure('Name', 'Imagem Bin 30%')
imshow(img_bin_2)
figure('Name', 'Imagem Bin 40%')
imshow(img_bin_3)
figure('Name', 'Imagem Bin 50%')
imshow(img_bin_4)
figure('Name', 'Imagem Bin 60%')
imshow(img_bin_5)

figure('Name', 'Histograma do Bin 20%')
imhist(img_bin_1)
figure('Name', 'Histograma Bin 30%')
imhist(img_bin_2)
figure('Name', 'Histograma Bin 40%')
imhist(img_bin_3)
figure('Name', 'Histograma Bin 50%')
imhist(img_bin_4)
figure('Name', 'Histograma Bin 60%')
imhist(img_bin_5)

raw_img = imread('images/polem.bmp');
img_bin_6 = graythresh(raw_img)
figure('Name', 'Imagem com Graythresh')
imshow(img_bin_6)
figure('Name', 'Histograma Graythresh')
imhist(img_bin_6)
