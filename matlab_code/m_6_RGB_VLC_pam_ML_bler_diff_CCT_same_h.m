%===============================================================================
% # Author            :   ZDF
% # Created on        :
% # last modified     :   12/12/2019 Thu
% # Description       :
% # 1. RGB
%   (finally used in paper) bler performance comparision : MIMO-PAM (ML) vs.
%	constellation design (ML) vs. AE
%===============================================================================
clear;clc;
%===============================================================================
% parameters
%===============================================================================
random_symbol_num_per_color=1e6;
color_num = 3;
SNR_dB     =   [0 : 2 : 20];
M = 2; % modulation order
xi = 0.1;
InterferenceMatrix = [(1-xi)    xi             0
    xi      (1-2*xi)      xi
    0        xi          (1-xi)];

%??????
Xr=0.7006;Yr=0.2993;
Xg=0.1547;Yg=0.8059;
Xb=0.1440;Yb=0.0297;
A1=[Xr/Yr,Xg/Yg,Xb/Yb;1,1,1;(1-Xr-Yr)/Yr,(1-Xg-Yg)/Yg,(1-Xb-Yb)/Yb];
c=[0.0114;0.0052;0.0427];
%6500K [1.1408 ; 1.3226 ; 1]
%5700 [1.3284   ; 1.3912 ;   1.0000]
%5000K [1.6339 ;   1.6348 ;   1.0000]
%4000 [2.3762  ;  2.0844  ;  1.0000]
%2700 [6.1892  ;  3.9912  ;  1.0000]
%===============================================================================
% ber caculation
%===============================================================================
ber_all=[];
for i=[1 3 5]
    %******************************************
    if i ==1
        x0=0.313;y0=0.337;
        b_0=[x0/y0;1;(1-x0-y0)/y0];
        % ?????????? %??????????
        ratio_0=(A1^-1)*b_0;
        H = InterferenceMatrix * diag(ratio_0.*c);
    elseif  i==2
        H = InterferenceMatrix * diag([1.3284   ; 1.3912 ;   1.0000]);
    elseif  i==3
        x0=0.346;y0=0.359;
        b_0=[x0/y0;1;(1-x0-y0)/y0];
        % ??????????
        ratio_0=(A1^-1)*b_0;
        H = InterferenceMatrix * diag(ratio_0.*c);
    elseif  i==4
        H = InterferenceMatrix * diag([2.3762  ;  2.0844  ;  1.0000]);
    elseif  i==5
        x0=0.459;y0=0.412;
        b_0=[x0/y0;1;(1-x0-y0)/y0];
        % ??????????
        ratio_0=(A1^-1)*b_0;
        H = InterferenceMatrix * diag(ratio_0.*c);
    end
    %****************parameter init*************************
    % H = InterferenceMatrix * diag([1.1408 ; 1.3226 ; 1]./[0.0114; 0.0052; 0.0427]);
    
    %initialize
    ber = zeros(1,length(SNR_dB));
    for i_p_tx=1:length(SNR_dB)
        SNR = 10.0 ^ (SNR_dB(i_p_tx) / 10.0);
        P_noise = 1/SNR;
        
        num_loop = 10;
        ber_sum = 0;
        for mm = 1:num_loop
            %===============================================================================
            % gen random bits
            %===============================================================================
            u_data = randi([0 M-1],color_num,random_symbol_num_per_color);
            modData = real(pammod(u_data,M));
            
            rec_data = H * modData;
            rec_data_normalize = rec_data/sqrt(mean(mean(rec_data.^2)));
            rec_data_with_noise = rec_data_normalize + sqrt(P_noise)*randn(size(rec_data));
            
            rec_data_with_noise = rec_data_with_noise*sqrt(mean(mean(rec_data.^2)));
            for t = 1:random_symbol_num_per_color
                r = rec_data_with_noise(:,t);
                d000 = norm( H*real(pammod([0;0;0],M))-r ) ;
                d001 = norm( H*real(pammod([0;0;1],M))-r ) ;
                d010 = norm( H*real(pammod([0;1;0],M))-r ) ;
                d011 = norm( H*real(pammod([0;1;1],M))-r ) ;
                d100 = norm( H*real(pammod([1;0;0],M))-r ) ;
                d101 = norm( H*real(pammod([1;0;1],M))-r ) ;
                d110 = norm( H*real(pammod([1;1;0],M))-r ) ;
                d111 = norm( H*real(pammod([1;1;1],M))-r ) ;
                min_d=min([d000 d001 d010 d011 d100 d101 d110 d111 ]')';
                if(min_d==d000)
                    y=[0;0;0];
                elseif(min_d==d001)
                    y=[0;0;1];
                elseif(min_d==d010)
                    y=[0;1;0];
                elseif(min_d==d011)
                    y=[0;1;1];
                elseif(min_d==d100)
                    y=[1;0;0];
                elseif(min_d==d101)
                    y=[1;0;1];
                elseif(min_d==d110)
                    y=[1;1;0];
                else
                    y=[1;1;1];
                end
                ber_sum = ber_sum + (sum( y~=u_data(:,t))>0);
            end
        end
        % bit error num / loop num / group num
        ber(i_p_tx)=ber_sum/random_symbol_num_per_color/num_loop;
        %**********************************************************************
    end
    ber_all = [ber_all;ber];
end
%===============================================================================
% results visualization
%===============================================================================
figure;
% MIMO-PAM
semilogy(SNR_dB, ber_all(1,:), 'b--o');
hold on;
% semilogy(SNR_dB, ber_all(2,:), 'b-*');
semilogy(SNR_dB, ber_all(2,:), 'b--s');
% semilogy(SNR_dB, ber_all(4,:), 'b--');
semilogy(SNR_dB, ber_all(3,:), 'b--x');

% constellation design
ber_all_constellation_design = ...
    [0.441034000000000,0.321632000000000,0.198122000000000,0.0925370000000000,0.0284110000000000,0.00454400000000000,0.000229000000000000,1.00000000000000e-06,0,0,0;
     0.457658000000000,0.342406000000000,0.219377000000000,0.111012000000000,0.0386510000000000,0.00754200000000000,0.000609000000000000,6.00000000000000e-06,0,0,0;
     0.536362700000000,0.439794200000000,0.327603800000000,0.211499600000000,0.108743900000000,0.0393097000000000,0.00826300000000000,0.000730000000000000,1.92000000000000e-05,2.00000000000000e-07,0];
 %[0.536362700000000,0.439794200000000,0.327603800000000,0.211499600000000,0.108743900000000,0.0393097000000000,0.00826300000000000,0.000730000000000000,1.92000000000000e-05,2.00000000000000e-07,0]
 %[0.534041100000000,0.436263600000000,0.323214600000000,0.205345700000000,0.102681500000000,0.0353534000000000,0.00683620000000000,0.000532400000000000,9.80000000000000e-06,1.00000000000000e-07,0]
semilogy(SNR_dB, ber_all_constellation_design(1,:), 'k-.o');
semilogy(SNR_dB, ber_all_constellation_design(2,:), 'k-.s');
semilogy(SNR_dB, ber_all_constellation_design(3,:), 'k-.x');

% AE
semilogy(SNR_dB, [4.38292003e-01 3.18066001e-01 1.94413999e-01 9.12490003e-02...
    2.86379999e-02 4.97100004e-03 3.37000002e-04 5.99999985e-06...
    0.00000000e+00 0.00000000e+00 0.00000000e+00], 'r-o');
semilogy(SNR_dB, [4.52737999e-01 3.35589001e-01 2.12667999e-01 1.05505999e-01...
    3.55420001e-02 6.64800000e-03 5.38000002e-04 9.99999975e-06...
    0.00000000e+00 0.00000000e+00 0.00000000e+00], 'r-s');
% semilogy(SNR_dB, [5.17010999e-01 4.15263000e-01 2.99620998e-01 1.84629001e-01...
%     8.93350005e-02 3.08679998e-02 6.62599998e-03 7.25999998e-04...
%     2.09999995e-05 0.00000000e+00 0.00000000e+00], 'r-x');
semilogy(SNR_dB, [5.21452808e-01 4.22432196e-01 3.09863901e-01 1.96285699e-01...
 1.00057801e-01 3.73122998e-02 8.92629996e-03 1.11669999e-03...
 5.40000001e-05 4.99999999e-07 0.00000000e+00], 'r-x');
% title('RGB')

% 5.21452808e-01 4.22432196e-01 3.09863901e-01 1.96285699e-01
%  1.00057801e-01 3.73122998e-02 8.92629996e-03 1.11669999e-03
%  5.40000001e-05 4.99999999e-07 0.00000000e+00

ylim([0.000006,1]);
grid on
legend('baseline 1 (6500K)','baseline 1 (5000K)','baseline 1 (2700K)','baseline 2 (6500K)','baseline 2 (5000K)','baseline 2 (2700K)','proposed scheme (6500K)','proposed scheme (5000K)','proposed scheme (2700K)')
xlabel('SNR (dB)')
ylabel('BLER')
% save workspace
save('RGB_same_h.mat')
