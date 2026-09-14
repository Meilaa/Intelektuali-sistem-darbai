%% Reading apple images

A1 = imread('apple_04.jpg');
A2 = imread('apple_05.jpg');
A3 = imread('apple_06.jpg');
A4 = imread('apple_07.jpg');
A5 = imread('apple_11.jpg');
A6 = imread('apple_12.jpg');
A7 = imread('apple_13.jpg');
A8 = imread('apple_17.jpg');
A9 = imread('apple_19.jpg');

%% Reading pear images

P1 = imread('pear_01.jpg');
P2 = imread('pear_02.jpg');
P3 = imread('pear_03.jpg');
P4 = imread('pear_09.jpg');

%% Calculate features for all images
% x1 = colour
% x2 = roundness

% Apples

hsv_value_A1 = spalva_color(A1);
metric_A1 = apvalumas_roundness(A1);

hsv_value_A2 = spalva_color(A2);
metric_A2 = apvalumas_roundness(A2);

hsv_value_A3 = spalva_color(A3);
metric_A3 = apvalumas_roundness(A3);

hsv_value_A4 = spalva_color(A4);
metric_A4 = apvalumas_roundness(A4);

hsv_value_A5 = spalva_color(A5);
metric_A5 = apvalumas_roundness(A5);

hsv_value_A6 = spalva_color(A6);
metric_A6 = apvalumas_roundness(A6);

hsv_value_A7 = spalva_color(A7);
metric_A7 = apvalumas_roundness(A7);

hsv_value_A8 = spalva_color(A8);
metric_A8 = apvalumas_roundness(A8);

hsv_value_A9 = spalva_color(A9);
metric_A9 = apvalumas_roundness(A9);

% Pears

hsv_value_P1 = spalva_color(P1);
metric_P1 = apvalumas_roundness(P1);

hsv_value_P2 = spalva_color(P2);
metric_P2 = apvalumas_roundness(P2);

hsv_value_P3 = spalva_color(P3);
metric_P3 = apvalumas_roundness(P3);

hsv_value_P4 = spalva_color(P4);
metric_P4 = apvalumas_roundness(P4);

%% TRAINING DATA
% First 5 objects are used for training
% A1, A2, A3 = apples
% P1, P2 = pears

% Feature 1 - colour
x1 = [hsv_value_A1 hsv_value_A2 hsv_value_A3 ...
      hsv_value_P1 hsv_value_P2];

% Feature 2 - roundness
x2 = [metric_A1 metric_A2 metric_A3 ...
      metric_P1 metric_P2];

% Desired output for training
% Apples = 1
% Pears = -1

T1 = [1 1 1 -1 -1];

%% TESTING DATA
% Remaining 8 objects are used only for testing
% A4-A9 = apples
% P3-P4 = pears

% Feature 1 - colour
x3 = [hsv_value_A4 hsv_value_A5 hsv_value_A6 ...
      hsv_value_A7 hsv_value_A8 hsv_value_A9 ...
      hsv_value_P3 hsv_value_P4];

% Feature 2 - roundness
x4 = [metric_A4 metric_A5 metric_A6 ...
      metric_A7 metric_A8 metric_A9 ...
      metric_P3 metric_P4];

% Desired output for testing
T = [1 1 1 1 1 1 -1 -1];

%% Generate random initial values

w1 = randn(1);
w2 = randn(1);
b = randn(1);

%% Training

eta = 0.1;
e = 1;

epoch = 0;

while e ~= 0

    epoch = epoch + 1;

    fprintf('\n========== EPOCH %d ==========\n', epoch);

    %% TRAIN

    for n = 1:5

        % Weighted sum
        v = x1(n)*w1 + x2(n)*w2 + b;

        % Perceptron output
        if v > 0
            y = 1;
        else
            y = -1;
        end

        % Instantaneous error
        e_n = T1(n) - y;

        % Display training information
        fprintf('Training: object %d, desired = %d, output = %d, error = %d\n', ...
                n, T1(n), y, e_n);

        % Update parameters
        w1 = w1 + eta*e_n*x1(n);
        w2 = w2 + eta*e_n*x2(n);
        b = b + eta*e_n;

    end

    %% TEST

    e = 0;

    fprintf('\n--- TESTING REMAINING OBJECTS ---\n');

    for n = 1:8

        % Weighted sum
        v = x3(n)*w1 + x4(n)*w2 + b;

        % Perceptron output
        if v > 0
            y = 1;
        else
            y = -1;
        end

        % Error
        e_n = T(n) - y;

        % Add to total error
        e = e + abs(e_n);

        % Display test result
        fprintf('Test: object %d, desired = %d, output = %d, error = %d\n', ...
                n, T(n), y, e_n);

    end

    fprintf('TOTAL TEST ERROR = %d\n', e);

end

%% Training finished

disp(' ');
disp('==============================');
disp('Training finished');
disp('==============================');

disp(['w1 = ', num2str(w1)]);
disp(['w2 = ', num2str(w2)]);
disp(['b  = ', num2str(b)]);

%% Final testing

disp(' ');
disp('==============================');
disp('FINAL TESTING');
disp('==============================');

for n = 1:8

    v = x3(n)*w1 + x4(n)*w2 + b;

    if v > 0
        y = 1;
    else
        y = -1;
    end

    fprintf('Test object %d: desired = %d, output = %d\n', ...
            n, T(n), y);

end