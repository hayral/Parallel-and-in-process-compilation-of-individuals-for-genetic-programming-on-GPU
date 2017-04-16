load alldata+nvcc
% figure 10 , 12 , 14



datax = dataMULnvcc;
data1 = dataMULnvrtc; 
data2 = dataMULnvrtc2;
data4 = dataMULnvrtc4;
data6 = dataMULnvrtc6;
data8 = dataMULnvrtc8;
% 
%  datax = dataK6nvcc;
%  data1 = dataK6nvrtc; 
%  data2 = dataK6nvrtc2;
%  data4 = dataK6nvrtc4;
%  data6 = dataK6nvrtc6;
%  data8 = dataK6nvrtc8;

%datax = datasearchnvcc;
%data1 = datasearchnvrtc; 
%data2 = datasearchnvrtc2;
%data4 = datasearchnvrtc4;
%data6 = datasearchnvrtc6;
%data8 = datasearchnvrtc8;


hold on

baselinedata = data1;


[p,v,e] = parseDataTable(baselinedata,2);
[p2,v2,e2] = parseDataTable(baselinedata,3);

[p3,v3,e] = parseDataTable(data1,2);
[p4,v4,e2] = parseDataTable(data1,3);

plot(p, (v+v2) ./ (v3+v4) ,'-+');


[p,v,e] = parseDataTable(baselinedata,2);
[p2,v2,e2] = parseDataTable(baselinedata,3);

[p3,v3,e] = parseDataTable(data2,2);
[p4,v4,e2] = parseDataTable(data2,3);

plot(p, (v+v2) ./ (v3+v4) ,'-+');


[p,v,e] = parseDataTable(baselinedata,2);
[p2,v2,e2] = parseDataTable(baselinedata,3);

[p3,v3,e] = parseDataTable(data4,2);
[p4,v4,e2] = parseDataTable(data4,3);

plot(p, (v+v2) ./ (v3+v4) ,'-+');



[p,v,e] = parseDataTable(baselinedata,2);
[p2,v2,e2] = parseDataTable(baselinedata,3);

[p3,v3,e] = parseDataTable(data6,2);
[p4,v4,e2] = parseDataTable(data6,3);

plot(p, (v+v2) ./ (v3+v4) ,'-+');


[p,v,e] = parseDataTable(baselinedata,2);
[p2,v2,e2] = parseDataTable(baselinedata,3);

[p3,v3,e] = parseDataTable(data8,2);
[p4,v4,e2] = parseDataTable(data8,3);

plot(p, (v+v2) ./ (v3+v4) ,'-+');





xlabel('population size (#individuals)');
    ylabel('speed up ratio');


T = get(gca,'tightinset');
set(gca,'position',[T(1) T(2) 1-T(1)-T(3) 1-T(2)-T(4)]); 
legend('inprocess','2 service processes','4 service processes','6 service processes','8 service processes','Location','northwest')

axis([0 320 0 8]);