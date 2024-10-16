function [bin_hight,bin_center]=pdf_plotter_new(N_vess,PDF_array)
% PDF_array is the data array is the data array which we want to plot the
% PDF of it.
[N,bin_center]=hist(PDF_array,N_vess);
% bar(bin_center,N);
bin_hight = N./sum(N)./(bin_center(2)-bin_center(1));