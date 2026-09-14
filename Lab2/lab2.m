clc
clear all
close all

%% Pradiniai duomenys

x = 0.1:1/22:1;

d = ((1 + 0.6*sin(2*pi*x/0.7)) + 0.3*sin(2*pi*x))/2;

figure
plot(x, d, '*');
hold on

%% Tinklo inicializavimas

% Pirmas sluoksnis - 5 neuronai
w11_1 = rand(1);
w21_1 = rand(1);
w31_1 = rand(1);
w41_1 = rand(1);
w51_1 = rand(1);

b1_1 = rand(1);
b2_1 = rand(1);
b3_1 = rand(1);
b4_1 = rand(1);
b5_1 = rand(1);

% Antras sluoksnis - vienas isejimo neuronas
w11_2 = rand(1);
w12_2 = rand(1);
w13_2 = rand(1);
w14_2 = rand(1);
w15_2 = rand(1);

b1_2 = rand(1);

%% Mokymo parametrai

eta = 0.1;

%% Tinklo mokymas - Backpropagation

for iter = 1:10000

    for i = 1:length(x)

        %% Feedforward

        % Pirmo sluoksnio pasvertos sumos
        v1_1 = x(i) * w11_1 + b1_1;
        v2_1 = x(i) * w21_1 + b2_1;
        v3_1 = x(i) * w31_1 + b3_1;
        v4_1 = x(i) * w41_1 + b4_1;
        v5_1 = x(i) * w51_1 + b5_1;

        % Pirmo sluoksnio sigmoidines aktyvacijos funkcijos
        y1_1 = 1 / (1 + exp(-v1_1));
        y2_1 = 1 / (1 + exp(-v2_1));
        y3_1 = 1 / (1 + exp(-v3_1));
        y4_1 = 1 / (1 + exp(-v4_1));
        y5_1 = 1 / (1 + exp(-v5_1));

        % Antro sluoksnio pasverta suma
        v1_2 = y1_1 * w11_2 + ...
               y2_1 * w12_2 + ...
               y3_1 * w13_2 + ...
               y4_1 * w14_2 + ...
               y5_1 * w15_2 + b1_2;

        % Isejimo sluoksnio aktyvacijos funkcija - tiesine
        y1_2 = v1_2;

        % Tinklo isejimas
        y = y1_2;

        %% Klaida

        e = d(i) - y;

        %% Backpropagation

        % Isejimo sluoksnio delta
        delta1_2 = e;

        % Pasleptojo sluoksnio delta
        % Sigmoidines funkcijos isvestine:
        % y * (1 - y)

        delta1_1 = y1_1 * (1 - y1_1) * delta1_2 * w11_2;
        delta2_1 = y2_1 * (1 - y2_1) * delta1_2 * w12_2;
        delta3_1 = y3_1 * (1 - y3_1) * delta1_2 * w13_2;
        delta4_1 = y4_1 * (1 - y4_1) * delta1_2 * w14_2;
        delta5_1 = y5_1 * (1 - y5_1) * delta1_2 * w15_2;

        %% Atnaujinami antro sluoksnio svoriai

        w11_2 = w11_2 + eta * delta1_2 * y1_1;
        w12_2 = w12_2 + eta * delta1_2 * y2_1;
        w13_2 = w13_2 + eta * delta1_2 * y3_1;
        w14_2 = w14_2 + eta * delta1_2 * y4_1;
        w15_2 = w15_2 + eta * delta1_2 * y5_1;

        b1_2 = b1_2 + eta * delta1_2;

        %% Atnaujinami pirmo sluoksnio svoriai

        w11_1 = w11_1 + eta * delta1_1 * x(i);
        w21_1 = w21_1 + eta * delta2_1 * x(i);
        w31_1 = w31_1 + eta * delta3_1 * x(i);
        w41_1 = w41_1 + eta * delta4_1 * x(i);
        w51_1 = w51_1 + eta * delta5_1 * x(i);

        %% Atnaujinami pirmo sluoksnio bias

        b1_1 = b1_1 + eta * delta1_1;
        b2_1 = b2_1 + eta * delta2_1;
        b3_1 = b3_1 + eta * delta3_1;
        b4_1 = b4_1 + eta * delta4_1;
        b5_1 = b5_1 + eta * delta5_1;

    end

end

%% Tinklo testavimas po mokymo

x_new = 0.1:1/22:1;

Y = zeros(1, length(x_new));

for i = 1:length(x_new)

    % Pirmo sluoksnio pasvertos sumos
    v1_1 = x_new(i) * w11_1 + b1_1;
    v2_1 = x_new(i) * w21_1 + b2_1;
    v3_1 = x_new(i) * w31_1 + b3_1;
    v4_1 = x_new(i) * w41_1 + b4_1;
    v5_1 = x_new(i) * w51_1 + b5_1;

    % Pirmo sluoksnio sigmoidines aktyvacijos funkcijos
    y1_1 = 1 / (1 + exp(-v1_1));
    y2_1 = 1 / (1 + exp(-v2_1));
    y3_1 = 1 / (1 + exp(-v3_1));
    y4_1 = 1 / (1 + exp(-v4_1));
    y5_1 = 1 / (1 + exp(-v5_1));

    % Antro sluoksnio pasverta suma
    v1_2 = y1_1 * w11_2 + ...
           y2_1 * w12_2 + ...
           y3_1 * w13_2 + ...
           y4_1 * w14_2 + ...
           y5_1 * w15_2 + b1_2;

    % Tiesine aktyvacijos funkcija
    y1_2 = v1_2;

    % Tinklo rezultatas
    Y(i) = y1_2;

end

%% Rezultatu atvaizdavimas

plot(x_new, Y, 'r');

xlabel('x');
ylabel('y');

legend('Norimas atsakas', 'Perceptrono atsakas');

grid on