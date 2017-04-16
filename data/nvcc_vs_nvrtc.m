load alldata+nvcc
%figure 2


% choose one from below
data1 = datasearchnvcc; 
data2 = datasearchnvrtc;

%data1 = dataK6nvcc; 
%data2 = dataK6nvrtc; 

%data1 = dataMULnvcc; 
%data2 = dataMULnvrtc; 

perindividual=true;

hold on

[p,v,e] = parseDataTable(data1,2);
[p2,v2,e2] = parseDataTable(data1,3);
if perindividual
    errorbar(p2,(v+v2),(e+e2));
else
    errorbar(p2,.001*(v+v2).*p,.001*(e+e2).*p);
end


[p,v,e] = parseDataTable(data2,2);
[p2,v2,e2] = parseDataTable(data2,3);
if perindividual
    errorbar(p2,(v+v2),(e+e2));
else
    errorbar(p2,.001*(v+v2).*p,.001*(e+e2).*p);
end

xlabel('population size (#individuals)');
if perindividual
    ylabel('compile time per individual (ms)');
else
    ylabel('population compile time (sec)');
end
legend('out of process compilation','inprocess compilation')

T = get(gca,'tightinset');
set(gca,'position',[T(1) T(2) 1-T(1)-T(3) 1-T(2)-T(4)]); 

if perindividual
    axis([0 320 0 80])
else
axis([0 320 0 6])
end