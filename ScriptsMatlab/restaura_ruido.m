clear all
close(gcf)

% carregando a imagem

prompt = {'Enter the image name'};
dlgtitle = 'File Name';
definput = {'cameraman.tif'};
%opts.Interpreter = 'tex';
image_name = inputdlg(prompt,dlgtitle,[1 50],definput);

img = imread(image_name{1}) ;


% determinando a altura e a largura da imagem
[h l] = size(img) ;

% Gerando ru?do sal e pimenta na imagem com probabilidade menor do que p

prompt = {'Enter a value of noise probability (in %)'};
dlgtitle = 'Probability Value';
definput = {'10'};
opts.Interpreter = 'tex';
prob = inputdlg(prompt,dlgtitle,[1 50],definput,opts);

p = str2double(prob)/100;

img_noise = img ;

for i = 1 : h
    for j = 1 : l
        a = rand; 
        if a < 0.5
            a = rand ;
            if a < p
                img_noise(i,j) = 255 ;  % ru?do sal
            end
        else
            a = rand ;
            if a < p  
                img_noise(i,j) = 0 ;  % ru?do pimenta
            end
        end
    end
end


img_clean_median = img_noise ;

for i = 2 : h-1
    for j = 2 : l-1
        if (img_noise(i,j) == 0) || (img_noise(i,j) == 255)
            aux = img_noise(i-1:i+1, j-1:j+1) ;
            vetor = [ aux(1,:) aux(2,:) aux(3,:) ];  
            v_ordenado = sort(vetor);
            img_clean_median(i,j) = v_ordenado(5) ;
        end
    end
end


img_clean_mean = img_noise ;

for i = 2 : h-1
    for j = 2 : l-1
        %if (img_noise(i,j) == 0) || (img_noise(i,j) == 255)
            aux = img_noise(i-1:i+1, j-1:j+1) ;
            img_clean_mean(i,j) =  sum(sum(aux))/9.0 ;
        %end
    end
end

subplot(2,3,1); imshow(img) 
title('Imagem Original')
subplot(2,3,2); imshow(img_noise)
title('Imagem com ruido sal-pimenta')
subplot(2,3,3); imshow(img_clean_median)
title('Imagem Restaurada - My Median')
subplot(2,3,5); imshow(img_clean_mean)
title('Imagem Restaurada - My Mean')


K = medfilt2(img_noise);
subplot(2,3,4); imshow(K)
title('Imagem Restaurada - Median (medfilt2)')

localMean = imboxfilt(img_noise,3);
subplot(2,3,6); imshow(localMean)
title('Imagem Restaurada - Mean imboxfilt')



        
       