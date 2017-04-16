load alldata+nvcc


% choose one from below
data1 = datasearchnvcc; 
data2 = datasearchnvrtc;



hold on

[p,v,e] = parseDataTable(data1,2);
[p2,v2,e2] = parseDataTable(data1,3);

[p3,v3,e] = parseDataTable(data2,2);
[p4,v4,e2] = parseDataTable(data2,3);

plot(p, (v+v2) ./ (v3+v4) ,'-+');


data1 = dataK6nvcc; 
data2 = dataK6nvrtc; 

[p,v,e] = parseDataTable(data1,2);
[p2,v2,e2] = parseDataTable(data1,3);

[p3,v3,e] = parseDataTable(data2,2);
[p4,v4,e2] = parseDataTable(data2,3);

plot(p, (v+v2) ./ (v3+v4) ,'-+');


data1 = dataMULnvcc; 
data2 = dataMULnvrtc; 

[p,v,e] = parseDataTable(data1,2);
[p2,v2,e2] = parseDataTable(data1,3);

[p3,v3,e] = parseDataTable(data2,2);
[p4,v4,e2] = parseDataTable(data2,3);

plot(p, (v+v2) ./ (v3+v4) ,'-+');

legend('Search Problem speedup ratio','K6 Regression speedup ratio','5-bit Multiplier speedup ratio')



xlabel('population size (#individuals)');
    ylabel('speed up ratio');


T = get(gca,'tightinset');
set(gca,'position',[T(1) T(2) 1-T(1)-T(3) 1-T(2)-T(4)]); 

axis([0 320 0 3]);