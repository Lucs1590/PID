# T1
a = [16, 3, 2, 13;
 5, 10, 11, 8;
 9, 6, 7, 12;
 4, 15, 14, 1;]

b = [ 16 3 2 4
20 30 4 4
5 6 7 8]

c = [ 20,30 40;
50,40,10
80 20 15]

d = [ 1,2,3,4;5 6 7 8
9 9 9 9
8,8,8,8;
4 5,9 4]

# T2
whos;
AI = uint8(a); BI = uint16(b); CI = uint32(c); DS = single(d);

# T3
a(2,3)
a(:,3)
a(1:2,1:3)
ab = a
ab(:,4) = 0
a(end,end)
a(end,end-2)
a(2:end,end:-2:1)
a(:)
a(:)'

# T4
# get elements of line 1 and 3, at colunm 2 and 3, that is, elements into colunm
# 2 and 3 that be on line 1 and 3 of matrix a
A6 = a([1 3], [2 3]);
AL = logical([1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1]);
a(AL);
a(3,3) == a(11);
# sum all elements
S2 = sum(sum(a));

# T5
I = mat2gray(a);
IA = I
IB = mat2gray(b, [2,30]);
IC = mat2gray(c, [10,40]);
ID = mat2gray(d, [0,10]);

# T6
E = [1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25
1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25
1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25
1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25
1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25
1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25
1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25
1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25
1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25
1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25
1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25
1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25
1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25
1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25
1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25
1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25];
IE = mat2gray(E);
imshow(IE);

# E1
figure(1);
imshow(IE,[0 0.1]);
figure(2);
imshow(IE,[0.3,0.7]);
figure(3);
imshow(IE,[]);

F = im2uint8(IE);
figure(4);
imshow(F);

# E2
[X,map] = gray2ind(IE,25);
imshow(X,map);

[X,map] = gray2ind(IE);
imshow(X,map);

[X,map] = gray2ind(IE,255);
imshow(X);

# T_8
BE = im2bw(IE);
imshow(BE)
whos IE, whos BE

# E3
F = [ -0.5 -0.5 -0.2 0.2 0.9 1.2 1.5 ;
-0.8 0.7 0.72 0.2 0.9 1.2 1.5;
-0.5 0.5 0.2 0.2 0.9 1.2 1.5;
-0.5 0.45 0.2 0.2 0.9 1.2 1.5;
-0.5 -0.5 0.42 0.2 0.49 1.2 1.5];
im2uint8(F);
mat2gray(F);
im2bw(F, 0.3);

# T_9
f = imread('images/chestxray_gray.jpg');
size(f)
[M,N] = size(f);
whos f;
imshow(f);
g = imread('images/rose_gray.tif');
imshow(f), figure, imshow(g)
# At Octave we don't have pixval function, so a alternative is:
function btn_down (obj, evt)
  cp = get (gca, 'CurrentPoint');
  x = round (cp(1, 1));
  y = round (cp(1, 2));
  img = get (findobj (gca, "type", "image"), "cdata");
  img_v = NA;
  if (x > 0 && y > 0 && x <= columns (img) && y <= rows (img))
    img_v = squeeze (img(y, x, :));
  endif

  if (numel (img_v) == 3) # rgb image
    title(gca, sprintf ("(%i, %i) = %i, %i, %i", x, y, img_v(1), img_v(2), img_v(3)));
  elseif (numel (img_v) == 1) # gray image
    title(gca, sprintf ("(%i, %i) = %i", x, y, img_v));
  endif
endfunction
# set (gcf, 'WindowButtonDownFcn', @btn_down);

# E_4
h = imread('images/bubbles.jpg');
i = imread('bubbles5.jpg');
imshow(h), figure, imshow(i)

KH = imfinfo('images/bubbles.jpg')
image_bytes_h = KH.Width*KH.Height*KH.BitDepth/8;
compress_ratio_h = image_bytes_h/KH.FileSize;

KI = imfinfo('bubbles5.jpg')
image_bytes_i = KI.Width*KI.Height*KI.BitDepth/8;
compress_ratio_i = image_bytes_i/KI.FileSize;

# E_5
f = imread('images/circuit.jpg');
imfinfo images/circuit.jpg

# T_10
f = imread('images/rose_gray.tif');
imshow(f)
whos f

fp = f(end:-1:1,:);
fl = f(:,end:-1:1);
imshow(f), figure, imshow(fp), figure, imshow(fl)
fc = f(65:198, 65:198);
fs = f(1:2:end, 1:2:end);
imshow(f), figure, imshow(fc), figure, imshow(fs)
plot(f(132,:))

# T_11
Z = zeros(5,5)
whos Z
U = ones(3,3)
whos U
M = magic(4)
whos M
sum(M(:,1)), sum(M(3,:))
R1 = rand(4,4)
R2 = randn(4,4)