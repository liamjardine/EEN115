function retopology= reformat_topo(input_data)
    s=input_data(:,4);
    d=input_data(:,5);
    lengths=input_data(:,6);

    num_node=max(s);
    retopology=zeros(num_node);

    for i=1:length(lengths)
        nodea=s(i);
        nodeb=d(i);
        length_ab=lengths(i);
        retopology(nodea, nodeb) = length_ab;

    end
end
