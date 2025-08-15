library(openxlsx)
library(clusterProfiler)
library(ggplot2)
library(enrichplot)
library(GOplot)
library(DOSE)
library(stringr)
library("org.Hs.eg.db")
getwd()

setwd("E:/drug_repurposing/GSE_datesets_results/finished_R_DEG/GSE182321_result/DESeq2_rersult/top300/")
data <- read.csv('./deg_ppi_nodes_large.csv')

gene<-bitr(data$Gene,fromType = 'SYMBOL',toType = 'ENTREZID',OrgDb = org.Hs.eg.db) 


KEGG<-enrichKEGG( 
  gene$ENTREZID, 
  organism = "hsa", 
  keyType = "kegg", 
  pvalueCutoff = 0.05, 
  pAdjustMethod = "BH",
)

write.table(KEGG@result,file = './kegg_pathways.csv',sep=",",row.names = FALSE) 

p <- dotplot(KEGG,color='pvalue') 
p <- p + theme(axis.text.x = element_text(size = 32), 
               axis.text.y = element_text(size = 32), 
               axis.title.x = element_text(size =32), 
               legend.text = element_text(size=32), 
               legend.title = element_text(size=32), 
) + scale_x_continuous(breaks = c(0.000, 0.235), labels = c("0.000", "0.235")) 

 
ggsave(  
  filename = "dotplot_tmp.png", 
  width = 10,             
  height = 10,            
  units = "in", 
  dpi = 600              
)

KEGGplotIn<-KEGG[1:10,c(3,4,11,13)] 
KEGGplotIn$geneID <-str_replace_all(KEGGplotIn$geneID,'/',',')
names(KEGGplotIn)<-c('ID','Term','adj_pval','Genes') 

KEGGplotIn$Category = "KEGG"  
rownames(gene) <- gene$SYMBOL 


genedata<- read.csv('./DEGs_DESeq2.csv') 
genedata <- genedata[,c(8,2)] 
colnames(genedata) <- c('ID','logFC') 

ENTREZID <- as.character(gene[genedata$ID,'ENTREZID'])  
ENTREZID[is.na(ENTREZID)]=""  
genedata$ID2 <- ENTREZID 
genedata <- genedata[genedata['ID2'] != '',] 
genedata <- genedata[,c(3,2)] 
colnames(genedata) <- c('ID','logFC') 

circ<-GOplot::circle_dat(KEGGplotIn,genedata) 

chord<-chord_dat(data = circ,genes = genedata)
symbol <- gene[match(rownames(chord),gene$ENTREZID),'SYMBOL']
rownames(chord) <- symbol

p <- GOChord(
  data = chord,
  title = '',
  space = 0,
  limit = c(1,1),
  gene.order = 'logFC',
  gene.space = 0.25,
  gene.size = 20,
  lfc.col = c('red','white','blue'), 
  
  process.label = 15 
)


ggsave(
  filename = "chord.png",
  width = 30,
  height = 30,
  units = "in",
  dpi = 300
)


GOCircle(circ,table.legend='True') 

write.csv(KEGGplotIn,'table_circle.csv')

ggsave(
  filename = "circle_tmp.png", 
  width = 16,             
  height = 10,            
  units = "in",         
  dpi = 600              
)

data(EC)

GOCluster(circ, KEGGplotIn$Term)
p<-GOCluster(circ,KEGGplotIn$Term,clust.by='logFC',lfc.col=c('darkgoldenrod1','black','cyan1'))
p <- p + theme(
  text = element_text(size = 12),
  
  legend.direction = "vertical",
  legend.text = element_text(size = 12),
  
)
ggsave(
  filename = "gocluster_tmp.png", 
  width = 20 ,             
  height = 20,            
  units = "in",         
  dpi = 600              
)

