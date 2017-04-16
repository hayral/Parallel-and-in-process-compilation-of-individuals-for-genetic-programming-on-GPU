% figure  9 , 11 , 13

load alldata


%data1 = dataMULnvrtc; 
%data2 = dataMULnvrtc2;
%data4 = dataMULnvrtc4;
%data6 = dataMULnvrtc6;
%data8 = dataMULnvrtc8;
%data1 = dataK6nvrtc; 
%data2 = dataK6nvrtc2;
%data4 = dataK6nvrtc4;
%data6 = dataK6nvrtc6;
%data8 = dataK6nvrtc8;
data1 = datasearchnvrtc; 
data2 = datasearchnvrtc2;
data4 = datasearchnvrtc4;
data6 = datasearchnvrtc6;
data8 = datasearchnvrtc8;

perindividual=true;

[p,v,e] = parseDataTable(data1,2);
[p2,v2,e2] = parseDataTable(data1,3);

hold on
if perindividual
    plot(p2,(v+v2) ,'-+','MarkerSize',4);
else
    plot(p2,.001*(v+v2).*p, '-+','MarkerSize',4);
end


[p,v,e] = parseDataTable(data2,2);
[p2,v2,e2] = parseDataTable(data2,3);

if perindividual
    plot(p2,(v+v2), '-+','MarkerSize',4);
else
    plot(p2,.001*(v+v2).*p, '-+','MarkerSize',4);
end


[p,v,e] = parseDataTable(data4,2);
[p2,v2,e2] = parseDataTable(data4,3);

if perindividual
    plot(p2,(v+v2), '-+','MarkerSize',4);
else
    plot(p2,.001*(v+v2).*p, '-+','MarkerSize',4);
end


[p,v,e] = parseDataTable(data6,2);
[p2,v2,e2] = parseDataTable(data6,3);

if perindividual
    plot(p2,(v+v2), '-+','MarkerSize',4);
else
    plot(p2,.001*(v+v2).*p, '-+','MarkerSize',4);
end


[p,v,e] = parseDataTable(data8,2);
[p2,v2,e2] = parseDataTable(data8,3);

if perindividual
    plot(p2,(v+v2), '-+','MarkerSize',4);
else
    plot(p2,.001*(v+v2).*p ,'-+','MarkerSize',4);
end

xlabel('population size (#individuals)');
if perindividual
    ylabel('time (ms)');
else
    ylabel('time (sec)');
end

T = get(gca,'tightinset');
set(gca,'position',[T(1) T(2) 1-T(1)-T(3) 1-T(2)-T(4)]); 
if perindividual
    legend('inprocess','2 service processes','4 service processes','6 service processes','8 service processes')
axis([0 320 0 30 ])
else
    legend('inprocess','2 service processes','4 service processes','6 service processes','8 service processes','Location','northwest')
    axis([0 320 0 1.5 ]) %K6
end