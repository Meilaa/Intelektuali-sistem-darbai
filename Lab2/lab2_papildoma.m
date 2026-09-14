clc
clear all
close all

%% Pradiniai duomenys

x1 = 0:0.1:1;
x2 = 0:0.1:1;

[X1, X2] = meshgrid(x1, x2);

% Norimas pavirsius
D = (sin(pi*X1) + cos(pi*X2)) / 2;

figure
surf(X1, X2, D)
title('Norimas pavirsius')
xlabel('x1')
ylabel('x2')
zlabel('d')
grid on


%% Duomenu paruosimas mokymui

x1_train = X1(:);
x2_train = X2(:);
d = D(:);


%% Tinklo inicializavimas

% Pirmas sluoksnis - 5 neuronai
% Kiekvienas neuronas turi po 2 svorius,
% nes yra 2 iejimai

% 1 neuronas
w11_1 = rand(1);
w12_1 = rand(1);
b1_1 = rand(1);

% 2 neuronas
w21_1 = rand(1);
w22_1 = rand(1);
b2_1 = rand(1);

% 3 neuronas
w31_1 = rand(1);
w32_1 = rand(1);
b3_1 = rand(1);

% 4 neuronas
w41_1 = rand(1);
w42_1 = rand(1);
b4_1 = rand(1);

% 5 neuronas
w51_1 = rand(1);
w52_1 = rand(1);
b5_1 = rand(1);


%% Antras sluoksnis - vienas isejimo neuronas

w11_2 = rand(1);
w12_2 = rand(1);
w13_2 = rand(1);
w14_2 = rand(1);
w15_2 = rand(1);

b1_2 = rand(1);


%% Mokymo parametrai

eta = 0.1;


%% Tinklo mokymas - Backpropagation

for iter = 1:5000

    for i = 1:length(d)

        %% Feedforward

        % Pasverta suma pirmame sluoksnyje

        v1_1 = x1_train(i)*w11_1 + ...
               x2_train(i)*w12_1 + b1_1;

        v2_1 = x1_train(i)*w21_1 + ...
               x2_train(i)*w22_1 + b2_1;

        v3_1 = x1_train(i)*w31_1 + ...
               x2_train(i)*w32_1 + b3_1;

        v4_1 = x1_train(i)*w41_1 + ...
               x2_train(i)*w42_1 + b4_1;

        v5_1 = x1_train(i)*w51_1 + ...
               x2_train(i)*w52_1 + b5_1;


        %% Sigmoidine aktyvacijos funkcija

        y1_1 = 1 / (1 + exp(-v1_1));
        y2_1 = 1 / (1 + exp(-v2_1));
        y3_1 = 1 / (1 + exp(-v3_1));
        y4_1 = 1 / (1 + exp(-v4_1));
        y5_1 = 1 / (1 + exp(-v5_1));


        %% Isejimo sluoksnis

        v1_2 = y1_1*w11_2 + ...
               y2_1*w12_2 + ...
               y3_1*w13_2 + ...
               y4_1*w14_2 + ...
               y5_1*w15_2 + b1_2;

        % Tiesine aktyvacijos funkcija
        y1_2 = v1_2;

        % Tinklo isejimas
        y = y1_2;


        %% Klaida

        e = d(i) - y;


        %% Backpropagation

        % Isejimo sluoksnis
        delta1_2 = e;


        % Pasleptasis sluoksnis
        delta1_1 = y1_1*(1-y1_1)*delta1_2*w11_2;
        delta2_1 = y2_1*(1-y2_1)*delta1_2*w12_2;
        delta3_1 = y3_1*(1-y3_1)*delta1_2*w13_2;
        delta4_1 = y4_1*(1-y4_1)*delta1_2*w14_2;
        delta5_1 = y5_1*(1-y5_1)*delta1_2*w15_2;


        %% Atnaujinami isejimo sluoksnio svoriai

        w11_2 = w11_2 + eta*delta1_2*y1_1;
        w12_2 = w12_2 + eta*delta1_2*y2_1;
        w13_2 = w13_2 + eta*delta1_2*y3_1;
        w14_2 = w14_2 + eta*delta1_2*y4_1;
        w15_2 = w15_2 + eta*delta1_2*y5_1;

        b1_2 = b1_2 + eta*delta1_2;


        %% Atnaujinami pirmo sluoksnio svoriai

        % 1 neuronas
        w11_1 = w11_1 + eta*delta1_1*x1_train(i);
        w12_1 = w12_1 + eta*delta1_1*x2_train(i);
        b1_1 = b1_1 + eta*delta1_1;

        % 2 neuronas
        w21_1 = w21_1 + eta*delta2_1*x1_train(i);
        w22_1 = w22_1 + eta*delta2_1*x2_train(i);
        b2_1 = b2_1 + eta*delta2_1;

        % 3 neuronas
        w31_1 = w31_1 + eta*delta3_1*x1_train(i);
        w32_1 = w32_1 + eta*delta3_1*x2_train(i);
        b3_1 = b3_1 + eta*delta3_1;

        % 4 neuronas
        w41_1 = w41_1 + eta*delta4_1*x1_train(i);
        w42_1 = w42_1 + eta*delta4_1*x2_train(i);
        b4_1 = b4_1 + eta*delta4_1;

        % 5 neuronas
        w51_1 = w51_1 + eta*delta5_1*x1_train(i);
        w52_1 = w52_1 + eta*delta5_1*x2_train(i);
        b5_1 = b5_1 + eta*delta5_1;

    end
end


%% Tinklo testavimas

Y = zeros(size(D));

for i = 1:length(d)

    %% Pirmas sluoksnis

    v1_1 = x1_train(i)*w11_1 + ...
           x2_train(i)*w12_1 + b1_1;

    v2_1 = x1_train(i)*w21_1 + ...
           x2_train(i)*w22_1 + b2_1;

    v3_1 = x1_train(i)*w31_1 + ...
           x2_train(i)*w32_1 + b3_1;

    v4_1 = x1_train(i)*w41_1 + ...
           x2_train(i)*w42_1 + b4_1;

    v5_1 = x1_train(i)*w51_1 + ...
           x2_train(i)*w52_1 + b5_1;


    %% Sigmoidines aktyvacijos funkcijos

    y1_1 = 1 / (1 + exp(-v1_1));
    y2_1 = 1 / (1 + exp(-v2_1));
    y3_1 = 1 / (1 + exp(-v3_1));
    y4_1 = 1 / (1 + exp(-v4_1));
    y5_1 = 1 / (1 + exp(-v5_1));


    %% Isejimo sluoksnis

    v1_2 = y1_1*w11_2 + ...
           y2_1*w12_2 + ...
           y3_1*w13_2 + ...
           y4_1*w14_2 + ...
           y5_1*w15_2 + b1_2;

    y1_2 = v1_2;

    Y(i) = y1_2;

end


%% Rezultatai

figure

% Tikrasis pavirsius
h1 = surf(X1, X2, D);

hold on

% Neuroninio tinklo aproksimacija
h2 = mesh(X1, X2, Y);

% Tikrieji mokymo taskai
h3 = scatter3(X1(:), X2(:), D(:), 20, 'filled');

xlabel('x1')
ylabel('x2')
zlabel('y')

grid on
view(45,30)

%% Klaida

MSE = mean((d - Y(:)).^2);

title(['Pavirsiaus aproksimacija, MSE = ' num2str(MSE)])

legend([h1 h2 h3], ...
    'Tikrasis pavirsius', ...
    'Neuroninio tinklo aproksimacija', ...
    'Mokymo taskai');