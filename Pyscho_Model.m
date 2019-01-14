%%%%%%%%  EEE-598 Speech and Audio Processing & Perception %%%%%%%%%%%%%%
%%%%%%%%%%%%%% MPEG-1 psychoacoustic model 1 for simple audio compression %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc;
clear all;
close all;

Music_files = {'audio/Track 55.wav'}; % Q7Music Audio files

fs = 44100;
for i=1:1  
    frame_num=400;
    
    [y{i}, Fs{i}] = audioread(Music_files{i}); % Reading the files
    %% Step 1
    r1 = splnorm(y{i},Fs{i}); 
    %% Step 2
    [tonalm,tonal_bark,noise,noise_bark,tonal_bin,noise_bin] = tonal(r1,Fs{i},frame_num);
    %% Step 3
    [tonalm_t,tonal_bark_t,noise_t,noise_bark_t,tonal_dec, tbark_dec, noise_dec, nbark_dec, t_bin, n_bin] = thresh(tonalm,tonal_bark,noise,noise_bark,tonal_bin,noise_bin, Fs{i});
    %% Step 4
    [T_TM, T_NM, t_bark, n_bark,t,n] = indmask(tonal_dec,noise_dec,t_bin, n_bin, r1(1:256,frame_num));
    %% Step 5
    [TG, nfrewdrop] = globmaskthresh(T_TM,T_NM,r1(1:256,frame_num),t,n, t_bark,n_bark);
    %% Plotting 
    f=1:256;
    f_bark = freq2bark(f,0);
    %Tq2 = real(10*log10(tq(f)));
    Tq = tq(f); % Absolute Threhold for Quiet
%     figure;
%     plot(f,r1(1:257,60));
%     xlabel('Frequency (Hz)')
%     ylabel('SPL (dB)');
%     title('PSD- SPL Normalized')
%     
    figure;
    plot(f_bark,r1(1:256,frame_num));
    hold on;
    xlabel('Bark Frequency (z)')
    ylabel('SPL (dB)');
    title('Step1: PSD- SPL Normalized')
    hold on;
    plot(f_bark,Tq,'linestyle','--','Color','k');
    ylim([-50 150]);
    legend('Original Signal','Quiet Threshold')
    
    hold off;
 
    figure;
    plot(f_bark,Tq,'linestyle','--','Color','k');
    hold on;
    plot(f_bark,r1(1:256,frame_num));
    xlabel('Bark Frequency (z)')
    ylabel('SPL (dB)');
    title('Step2 :Tonal+ Noise Maskers')
    hold on;
    plot(tonal_bark,tonalm,'x','LineWidth',2,'DisplayName','Tonal masker');
    hold on;
    plot(noise_bark,noise,'o','LineWidth',2,'DisplayName','Noise masker');
    yLimits = get(gca,'YLim');  %# Get the range of the y axis
    for j=1:length(tonal_bark)
        line([tonal_bark(j) tonal_bark(j)],[tonalm(j) yLimits(1)],'linestyle',':','Color','r');
    end
    for j=1:length(noise_bark)
        line([noise_bark(j) noise_bark(j)],[noise(j) yLimits(1)],'linestyle',':','Color','g');
    end
    ylim([-50 150]);
    legend('Quiet Threshold','Original Signal','Tonal masker','Noise Masker')


    
%     hold off;
%     
%     figure;
%     plot(f_bark,r1(1:257,frame_num));
%     xlabel('Bark Frequency (z)')
%     ylabel('SPL (dB)');
%     title('Step1+Step2+Step3:PSD-SPL Normalized, Tonal+ Noise Maskers+ Threshold')
%     hold on;
%     plot(tonal_bark_t,tonalm_t,'x');
%     hold on;
%     plot(noise_bark_t,noise_t,'o');
    
    hold off;
    
    figure;
    plot(f_bark,Tq,'linestyle','--','Color','k');
    hold on;
    plot(f_bark,r1(1:256,frame_num));
    xlabel('Bark Frequency (z)')
    ylabel('SPL (dB)');
    title('Step3 :Tonal,Noise Maskers + Threshold + Decimation')
    hold on;
    plot(tbark_dec,tonal_dec,'x','LineWidth',2,'DisplayName','Tonal masker');
    hold on;
    plot(nbark_dec,noise_dec,'o','LineWidth',2,'DisplayName','Noise masker');
    yLimits = get(gca,'YLim');  %# Get the range of the y axis
    for j=1:length(tbark_dec)
        line([tbark_dec(j) tbark_dec(j)],[tonal_dec(j) yLimits(1)],'linestyle',':','Color','r');
    end
    for j=1:length(noise_dec)
        line([nbark_dec(j) nbark_dec(j)],[noise_dec(j) yLimits(1)],'linestyle',':','Color','g');
    end
    ylim([-50 150]);
    legend('Quiet Threshold','Original Signal','Tonal masker','Noise Masker')


    
    hold off;
    
    figure;
    plot(f_bark,Tq,'linestyle','--','Color','k');
    hold on;
    plot(f_bark,r1(1:256,frame_num),'linestyle',':');
    xlabel('Bark Frequency (z)')
    ylabel('SPL (dB)');
    title('Step4 :Tonal Maskers + Indiv. Tonal Maskers')
    hold on;
    plot(t_bark,tonal_dec,'x','LineWidth',2,'DisplayName','Tonal masker');
%     yLimits = get(gca,'YLim');  %# Get the range of the y axis
%     for j=1:length(tbark_dec)
%         line([tbark_dec(j) tbark_dec(j)],[tonal_dec(j) yLimits(1)],'linestyle','--','Color','r');
%     end
    hold on;
    plot(f_bark,T_TM);
    ylim([-50 150]); 
        legend('Quiet Threshold','Original Signal','Tonal masker')

    
    hold off;
    
    figure;
    plot(f_bark,Tq,'linestyle','--','Color','k','linestyle',':');
    hold on;
    plot(f_bark,r1(1:256,frame_num),'linestyle',':');
    xlabel('Bark Frequency (z)')
    ylabel('SPL (dB)');
    title('Step4 :Noise Maskers + Indiv. Noise Maskers')
    hold on;
    plot(n_bark,noise_dec,'o','LineWidth',2,'DisplayName','Noise masker');
%     yLimits = get(gca,'YLim');  %# Get the range of the y axis
%     for j=1:length(noise_dec)
%         line([nbark_dec(j) nbark_dec(j)],[noise_dec(j) yLimits(1)],'linestyle','--','Color','g');
%     end
    
    hold on;
    plot(f_bark,T_NM);
    ylim([-50 150]);
        legend('Quiet Threshold','Original Signal','Noise Masker')


    hold off;
    
    figure;
    plot(f_bark,Tq,'linestyle','--','Color','k');
    hold on;
    plot(f_bark,r1(1:256,frame_num),'linestyle',':');
    xlabel('Bark Frequency (z)')
    ylabel('SPL (dB)');
    title('Step5 :Global Masking Threshold + Tonal & Noise Maskers ')
    hold on;
    plot(t_bark,tonal_dec,'x','LineWidth',2,'DisplayName','Noise masker');
    hold on;
    plot(n_bark,noise_dec,'o','LineWidth',2,'DisplayName','Tonal masker');
    
%     yLimits = get(gca,'YLim');  %# Get the range of the y axis
%     for j=1:length(tbark_dec)
%         line([tbark_dec(j) tbark_dec(j)],[tonal_dec(j) yLimits(1)],'linestyle','--','Color','r');
%     end
%     for j=1:length(noise_bark)
%         line([nbark_dec(j) nbark_dec(j)],[noise_dec(j) yLimits(1)],'linestyle','--','Color','g');
%     end
    
    hold on;
    plot(f_bark,TG);
    ylim([-50 150]);
        legend('Quiet Threshold','Original Signal','Tonal masker','Noise Masker','Global Threshold')


    
end

%%
function snorm = splnorm(X,fs)
s = X(:,1);
n=512;
b = 16;
preemph = [1 -0.97];
s = filter(1,preemph,s);
% s=s-mean(s); % remove DC component
% x=s/max(abs(s));  %normalization
%x = s/(n*2^(b-1));
% t=length(x)/fs;
x=s;
N=floor(length(x)/512); %find how many samples will each frame contain
n_overlap_frames = floor((N*512-512)/480);
x_overlap_16 = zeros(512,n_overlap_frames);
x_hann = zeros(512,n_overlap_frames);
x_fft = zeros(512,n_overlap_frames);
psd = zeros(512,n_overlap_frames);
P = zeros(512,n_overlap_frames);
PN=90.302;
for k=0:n_overlap_frames
    x_overlap_16(:,k+1)=x(1+(n*k*15/16):n*(k+1)-((k*n)/16));
    x_hann(:,k+1)= hanning(length(x_overlap_16(:,k+1))).*x_overlap_16(:,k+1);
    x_fft(:,k+1)= fft(x_hann(:,k+1));
    psd(:,k+1)= (abs(x_fft(:,k+1)).^2);
    %psd(2:end-1,k+1)= 2*psd(2:end-1,k+1);
    P(:,k+1)= PN + 10*log10(psd(:,k+1));
end
snorm =  P;

end

%%
function [tone,fre,noise,kbar,tbin,nbin] = tonal(X,fs,frame_num)
len = length(X);
c=1;
%S = zeros(18,1);
S=[];
for k=frame_num:frame_num
    for j=1:256
        if j>2 && j<63
            if X(j,k)>X(j-1,k) &&  X(j,k)>X(j+1,k) && X(j,k)>(X(j-2,k)+7) && X(j,k)>(X(j+2,k)+7)    
                %S(c)=j;
                S = [S;j];
                c=c+1;
            end
        elseif j>62 && j<127
            if X(j,k)>X(j-1,k) &&  X(j,k)>X(j+1,k) && X(j,k)>X(j-3,k)+7 && X(j,k)>X(j+3,k)+7 && X(j,k)>(X(j-2,k)+7) && X(j,k)>(X(j+2,k)+7)  
                 S = [S;j];
                %disp(j)
                c=c+1;
            end
        elseif j>126 && j<257
             if X(j,k)>X(j-1,k) &&  X(j,k)>X(j+1,k) && X(j,k)>X(j-6,k)+7 && X(j,k)>X(j+6,k)+7 && X(j,k)>X(j-3,k)+7 && X(j,k)>X(j+3,k)+7 && X(j,k)>(X(j-2,k)+7) && X(j,k)>(X(j+2,k)+7)&& X(j,k)>X(j-4,k)+7 && X(j,k)>X(j+4,k)+7 && X(j,k)>(X(j-5,k)+7) && X(j,k)>(X(j+5,k)+7)    
                 S = [S;j];
                %disp(j)
                c=c+1;
             end
        end
    end
    %disp(k)
end


%tone=S;
[row, col]=size(S);
%disp(row)
P_TM= zeros(row,1);
%disp(S)
for k=frame_num:frame_num
    for j=1:row
        P_TM(j,1) = 10*log10(10^(0.1*X(S(j)-1,k))+10^(0.1*X(S(j),k))+10^(0.1*X(S(j)+1,k)));
    end
end
tone = P_TM;  
tbin = round(S);

tonal_bark = freq2bark(S,0);
fre = tonal_bark;
%disp(S)
k_bar = zeros(1,length(S));
for i=1:length(S)
    if i==1 || i==length(S)
        k_bar(i)= S(i);
    else
        k_bar(i) = geomean([S(i-1) S(i+1)]);
   end
end

%nbin=(k_bar);

%k_bark = freq2bark(nbin,0);
%kbar=k_bark;

%P_NM = zeros(length(k_bar),1);
kbar2=zeros(25,1);

kbar2(1)= geomean([0 100]);
kbar2(2)= geomean([100 200]);
kbar2(3)= geomean([200 300]);
kbar2(4)= geomean([300 400]);
kbar2(5)= geomean([400 510]);
kbar2(6)= geomean([510 630]);
kbar2(7)= geomean([630 770]);
kbar2(8)= geomean([770 920]);
kbar2(9)= geomean([920 1080]);
kbar2(10)= geomean([1080 1270]);
kbar2(11)= geomean([1270 1480]);
kbar2(12)= geomean([1480 1720]);
kbar2(13)= geomean([1720 2000]);
kbar2(14)= geomean([2000 2320]);
kbar2(15)= geomean([2320 2700]);
kbar2(16)= geomean([2700 3150]);
kbar2(17)= geomean([3150 3700]);
kbar2(18)= geomean([3700 4400]);
kbar2(19)= geomean([4400 5300]);
kbar2(20)= geomean([5300 6400]);
kbar2(21)= geomean([6400 7700]);
kbar2(22)= geomean([7700 9500]);
kbar2(23)= geomean([9500 12000]);
kbar2(24)= geomean([12000 15500]);
kbar2(25)= geomean([15500 22050]);

nbin = round(kbar2/44100*512);
nbin(nbin==0)=1;
%kbar = freq2bark(nbin,0);
kbar3 = zeros(length(S),1);

bw = [0 100 200 300 400 510 630 770 920 1080 1270 1480 1720 2000 2320 2700 3150 3700 4400 5300 6400 7700 9500 12000 15500 22050];
P_NM2 = zeros(length(S),1);
d=1;
S2 = (S*44100/512);
%disp(S2);
for k=frame_num:frame_num
    for j=1:length(bw)-1
        X2=X;
            u = round(bw(j+1)/44100*512);
            if j==1
                l=1;
            else
                l = round(bw(j)/44100*512);
            end
            %disp(d)
            
             %disp(bw(j+1))
             %disp(bw(j))

%             if l==1
%                 X2(l,k)=0;
%             elseif l==2
%                 X2(l-1:l+1,k)=0;
%             elseif l>2 && l<63
%                 X2(l-2:l+2,k)=0;
%             elseif l>62 && l<127
%                 X2(l-3:l+3,k)=0;
%             elseif l>126 && l<257
%                 X2(l-6:l+6,k)=0;
%             end
%         
%             if u==1
%                 X2(u,k)=0;
%             elseif u==2
%                 X2(u-1:u+1,k)=0;
%             elseif u>2 && u<63
%                 X2(u-2:u+2,k)=0;
%             elseif u>62 && u<127
%                 X2(u-3:u+3,k)=0;
%             elseif u>126 && u<257
%                 X2(u-6:u+6,k)=0;
%             end
            P_NM2(d,1) = 10*log10(sum(10.^(0.1*X2(l:u,k))));
            d=d+1;
        
    end
end

% disp(P_NM2)

d=1;
for k=frame_num:frame_num
    for j=1:length(bw)-1
        X2=X;
        if d>length(S2)
            break;
        end
        if S2(d)<bw(j+1)
           
            kbar3(d)=j;
           
%             disp(d)
%              disp(S2(d))
%              disp(bw(j+1))
%              disp(bw(j))
%         disp(l)
%         disp(u)

            d=d+1;
            j=1;
        end
    end
end

%disp(kbar3)
%disp(kbar)
kbar2(kbar3)=[];
nbin = round(kbar2/44100*512);
nbin(nbin==0)=1;
%disp(kbar)
kbar = freq2bark(nbin,0);

m=0;
% nbin = round(kbar3/44100*512);
% nbin(nbin==0)=1;
% kbar = freq2bark(nbin,0);

% for k=frame_num:frame_num
%     for j=1:length(k_bar)
%         X(S(j),k)=0;
%         X(S(j)-1,k)=0;
%         X(S(j)+1,k)=0;
%         if S(j)>2 && S(j)<63
%             X(S(j)-2,k)=0;
%             X(S(j)+2,k)=0;
%         elseif S(j)>62 && S(j)<127
%             X(S(j)-2,k)=0;
%             X(S(j)+2,k)=0;
%             X(S(j)-3,k)=0;
%             X(S(j)+3,k)=0;
%         elseif S(j)>126 && S(j)<257
%             X(S(j)-2,k)=0;
%             X(S(j)+2,k)=0;
%             X(S(j)-3,k)=0;
%             X(S(j)+3,k)=0;
%             X(S(j)-4,k)=0;
%             X(S(j)+4,k)=0;
%             X(S(j)-5,k)=0;
%             X(S(j)+5,k)=0;
%             X(S(j)-6,k)=0;
%             X(S(j)+6,k)=0;
%         end
%         P_NM(j,1) = 10*log10(sum(10.^(0.1*X(1:256,k))));
%     end
% end

% for k=frame_num:frame_num
%     for j=1:length(k_bar)
%         if j==1 || j==length(k_bar)
%             P_TM(j,k)=0;
%         else
%             P_TM(j,k)=0;
%             P_TM(j-1,k)=0;
%             P_TM(j+1,k)=0;
%             if S(j)>2 && S(j)<63
%                 P_TM(j-2,k)=0;
%                 P_TM(j+2,k)=0;
%             elseif S(j)>62 && S(j)<127
%                 P_TM(j-2,k)=0;
%                 P_TM(j+2,k)=0;
%                 P_TM(j-3,k)=0;
%                 P_TM(j+3,k)=0;
%             elseif S(j)>126 && S(j)<257
%                 P_TM(j-2,k)=0;
%                 P_TM(j+2,k)=0;
%                 P_TM(j-3,k)=0;
%                 P_TM(j+3,k)=0;
%                 P_TM(j-4,k)=0;
%                 P_TM(j+4,k)=0;
%                 P_TM(j-5,k)=0;
%                 P_TM(j+5,k)=0;
%                 P_TM(j-6,k)=0;
%                 P_TM(j+6,k)=0;
%             end
%         end
%         P_NM(j,1) = 10*log10(sum(10.^(0.1*P_TM(:,k))));
%     end
% end

P_NM2(kbar3)=[];
%kbar = freq2bark(kbar3,0);
noise = P_NM2;
%disp(noise)

end

%% 
function [tonal2, tb, noise2, nb,tonal3, tb2, noise3, nb2, tonal_bin2, noise_bin2 ] = thresh(tonalm,tonal_bark,noise,noise_bark, tonal_bin, noise_bin,fs)
len = length(tonalm);
% disp(tonalm)
%disp(len);
f=1:256;
Tq = tq(f);
%tonal4 = tonalm(tonalm>tq(tonal_bin));
%disp(tq(tonal_bin));
%disp(tonal4);
aa=0;
t=0;
n=0;
        
for i=1:len
    if tonalm(i)< Tq(tonal_bin(i))
        disp(i);
%         disp(tonalm(i))
%         disp(Tq(tonal_bin(i)))
        tonal_bark(i)=-5555;
        tonal_bin(i)=-5555;
        tonalm(i)=-5555;
        
       
        t=t+1;
    end
end
tonal_bark=tonal_bark(tonal_bark~=-5555);
tonal_bin=tonal_bin(tonal_bin~=-5555);
tonalm=tonalm(tonalm~=-5555);
len= length(tonalm);

% for i=1:len-1
%     if (tonal_bark(i+1)-tonal_bark(i)) <= 0.5 
% %          disp(tonal_bark(i+1))
% %         disp(tonal_bark(i))
%         tonal_bark(i)=-5555;
%         tonal_bin(i)=-5555;
%         tonalm(i)=-5555;
%       
%         t=t+1;
%     end
% end
% tonal_bark=tonal_bark(tonal_bark~=-5555);
% tonal_bin=tonal_bin(tonal_bin~=-5555);
% tonalm=tonalm(tonalm~=-5555);

%disp(length(noise_bin));
for i=1:length(noise_bin)
    if noise(i)< Tq(noise_bin(i))
%            disp(noise(i))
%         disp(Tq(noise_bin(i)))
        noise_bark(i)=-5555;
        noise_bin(i)=-5555;
        noise(i)=-5555;
     
        n=n+1;
    end
end
noise_bark=noise_bark(noise_bark~=-5555);
noise_bin=noise_bin(noise_bin~=-5555);
noise=noise(noise~=-5555);

combo = [tonal_bark, noise_bark];
combo2 = [tonal_bin.', noise_bin.'];
combo3 = [tonalm.', noise.']
[sorted, ind] = sort(combo);
for i=1:length(sorted)-1
    if sorted(i+1)-sorted(i)<=0.5
        %sorted(i)=-5555;
        ind(i)=-5555;
        %combo(i)=-5555;
        %ind(i)=-5555;
    end
end
ind=ind(ind~=-5555);
%combo=combo(combo~=-5555);
ind_t = ind(ind<=length(tonalm));
ind_n = ind(ind>length(tonalm));
tb = combo(ind_t);
nb = combo(ind_n);
tonal_bin = combo2(ind_t);
noise_bin = combo2(ind_n);
tonal2 = combo3(ind_t);
noise2 = combo3(ind_n);



% for i=1:length(noise_bin)-1
%     if noise_bark(i+1)-noise_bark(i) <= 0.5 
%         %disp(i)
% %          disp(noise_bark(i+1))
% %         disp(noise_bark(i))
%         noise_bark(i)=-5555;
%         noise_bin(i)=-5555;
%         noise(i)=-5555;
%         n=n+1;
%         
%     end
% end
% noise_bark=noise_bark(noise_bark~=-5555);
% noise_bin=noise_bin(noise_bin~=-5555);
% noise=noise(noise~=-5555);


% tonal2 = tonalm(tonalm~=-5555);
% tb = tonal_bark(tonal_bark~=-5555);
% 
% noise2 = noise(noise~=-5555);
% 
% nb = noise_bark(noise_bark~=-5555);


%disp(size(noise2))

% Subsampling
%i_tonal = zeros(1,len);
% tonal_bin = tonal_bin(tonal_bin~=0);
% noise_bin = noise_bin(noise_bin~=0);
%disp(freq2bark(tonal_bin,0))
c=1;
i_tonal=[];
i_noise=[];

for i=1:length(tonal_bin)
    if tonal_bin(i)>=1 && tonal_bin(i)<=48
        i_tonal(c)=tonal_bin(i);
        c=c+1;
    elseif tonal_bin(i)>=49 && tonal_bin(i)<=96
        i_tonal(c)=tonal_bin(i)+ mod(tonal_bin(i),2);
        c=c+1;
    elseif tonal_bin(i)>=97 && tonal_bin(i)<=232
        i_tonal(c)=tonal_bin(i)+ 3 -( mod(tonal_bin(i)-1,4));
        c=c+1;
    else 
        tonal2(i)=-1;
    end
end
% disp(tonal2)
% disp(tonal_bin)
% disp(i_tonal)
%i_noise = zeros(1,len);
%disp(noise_bin);
c=1;

for i=1:length(noise_bin)
    if noise_bin(i)>=1 && noise_bin(i)<=48
        i_noise(c)=noise_bin(i);
        c=c+1;
    elseif noise_bin(i)>=49 && noise_bin(i)<=96
        i_noise(c)=noise_bin(i)+ mod(noise_bin(i),2);
        c=c+1;
    elseif noise_bin(i)>=97 && noise_bin(i)<=232
        i_noise(c)=noise_bin(i)+ 3 -( mod(noise_bin(i)-1,4));
        c=c+1;
    else 
        noise2(i)=-1;
    end
end
i_tonal = round(i_tonal);
i_noise= round(i_noise);
% disp(i_noise)
tonal_bin2=i_tonal;
noise_bin2=i_noise;

t_bark = freq2bark(i_tonal,0);
%disp(t_bark)
tonal3 = tonal2(tonal2~=-1);
% disp(length(tonal2))

tb2 = t_bark;
% disp(tb)
% disp(tb2)
n_bark = freq2bark(i_noise,0);
noise3 = noise2(noise2~=-1);
%disp(size(noise3))
nb2 = n_bark;
end

%% 

function b = freq2bark(freq_bins,flag)
fs = 44100;
freq_arr = fs*freq_bins/512;
bark = zeros(1,length(freq_arr));
for i=1:length(bark)
    if freq_arr(i)<=1500
        bark(i)=13*atan(0.76*freq_arr(i)/1000) + 3.5*atan((freq_arr(i)/7500).^2);
    else
        bark(i)=8.7 + 14.2*log10(freq_arr(i)/1000);
        
    end
end
if flag==1
    b = round(bark);
else
    b= (bark);
end
end

%%
function ath = tq(f_bin)
f = f_bin*44100/512;
th = 3.64*((f/1000).^(-0.8))-6.5*exp(-0.6*((f/1000)-3.3).^2) + 0.010*((f/1000).^4) ;
ath = th;
end
%%
function s = SF(i,j,mask,bin,spl)
delta_z = freq2bark(i,0)-freq2bark(bin(j),0);
s1=0;
%disp(bin)
if delta_z >=-3 && delta_z <-1
    s1 = 17*delta_z - 0.4*mask(j) + 11;
elseif delta_z >=-1 && delta_z <0
    s1 = (0.4*mask(j)+6)*delta_z;
elseif delta_z >=0 && delta_z <1
    s1 = -17*delta_z;
elseif delta_z >=1 && delta_z <8
    %s1 = (0.15*spl(bin(j))-17)*delta_z - 0.15*spl(bin(j));
    s1 = (0.15*mask(j)-17)*delta_z - 0.15*mask(j);
end

s=s1;

end
%%
function [a,b,t_bark,n_bark,t_bin,n_bin] = indmask(tonal_dec,noise_dec,t_bin, n_bin,spl)
T_TM = zeros(256,length(t_bin));
T_NM = zeros(256,length(n_bin));
t_bark = freq2bark(t_bin,0);
bw = [0 100 200 300 400 510 630 770 920 1080 1270 1480 1720 2000 2320 2700 3150 3700 4400 5300 6400 7700 9500 12000 15500 22050];

for j=1:256
    for k=1:length(t_bin)
        m= round(t_bark(k));
             u = round(bw(m+1)/44100*512);
            
             l = round(bw(m)/44100*512);
        T_TM(j,k) = tonal_dec(k) - 0.275*freq2bark(t_bin(k),0) + SF(j,k,tonal_dec,t_bin,spl)-6.025;
        %T_TM(1:l-1,k)=0;
        %T_TM(u+1:j,k)=0;
    end
end
n_bark = freq2bark(n_bin,0);

for j=1:256
    for k=1:length(n_bin)
              m= round(n_bark(k));
              
             u = round(bw(m+1)/44100*512);
            
             l = round(bw(m)/44100*512);
            
        T_NM(j,k) = noise_dec(k) - 0.175*freq2bark(n_bin(k),0) + SF(j,k,noise_dec,n_bin,spl)-2.025;
        %disp(l)
        %disp(u)
        %T_NM(1:l-1,k)=0;
        %T_NM(u+1:j,k)=0;
    end
end

a=T_TM;
b=T_NM;
end
%%
function [tg, nfreqd] = globmaskthresh(T_TM,T_NM,X,t_bin,n_bin, t_bark, n_bark)
gmt = zeros(256,1);
f = 1:256;
Tq = tq(f);
n=0;

bw = [0 100 200 300 400 510 630 770 920 1080 1270 1480 1720 2000 2320 2700 3150 3700 4400 5300 6400 7700 9500 12000 15500 22050];

for j=1:256
    for k=1:length(t_bin)
        m= round(t_bark(k));
             u = round(bw(m+1)/44100*512);
            
             l = round(bw(m)/44100*512);
        T_TM(1:l-1,k)=0;
        T_TM(u+1:j,k)=0;
    end
end
n_bark = freq2bark(n_bin,0);

for j=1:256
    for k=1:length(n_bin)
              m= round(n_bark(k));
              
             u = round(bw(m+1)/44100*512);
            
             l = round(bw(m)/44100*512);
            
        %disp(l)
        %disp(u)
        T_NM(1:l-1,k)=0;
        T_NM(u+1:j,k)=0;
    end
end
for i=1:256
       
    gmt(i)= 10*log10(10.^(0.1.*Tq(i))+ sum(10.^(0.1.*(T_TM(i,:)))) + sum(10.^(0.1.*(T_NM(i,:)))));
    if gmt(i)> X(i)
        n=n+1;
    end
end
    
tg = gmt;    
nfreqd = n;

end
