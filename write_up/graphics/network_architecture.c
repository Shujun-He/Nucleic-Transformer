digraph G{
    graph [ dpi = 800]
    rankdir="TB";
    size="10,3";
    DNA[shape=box,label="Raw DNA sequences"];
	conv[shape=box,label="1D convolutions"];
	kmers[shape=box,label="Sequences of k-mers"];
	concat[shape=box,label="Concatenate all k-mer sequences"];
	transformer[shape=box,label="Transformer encoder"];
	recon[shape=box,label="Error correction"];
	error[shape=box,label="Erroneous positions classification"];
	pooling[shape=box,label="Pooling"];
	output1[shape=box,label="Promoter classification"]
	//output2[shape=box,label="Decoders for\nerror correction\n& sequence reconstruction"]
	//output3[shape=box,label="Decoders for\npredicting\nerroneous positions"]
	DNA->conv;
	conv->kmers;
	kmers->concat;
	concat->transformer;
	transformer->pooling->output1;
	transformer->recon;
	transformer->error;
	
/* 	transformer->output2;
	transformer->output3; */
}