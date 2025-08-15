# # Ensure BiocManager is installed
# if(!requireNamespace("BiocManager", quietly = TRUE))
#   install.packages("BiocManager")
getwd()
setwd("E:/drug_repurposing/GSE_datesets_results/finished_R_DEG/GSE174409_result/")
# Load required libraries
library(data.table)  
library(dplyr)       
library(ggplot2)     
#library(pheatmap)    
library(DESeq2)      
library(GEOquery)
library(biomaRt)
library(AnnotationDbi)
library(org.Hs.eg.db)
library(clusterProfiler)
# library(edgeR)       
# library(limma)      
#library(tinyarray)   

# Download data from GEO
eSet <- getGEO("GSE174409", destdir = ".", getGPL = T) 
#exprSet <- fread("E:/drug_repurposing/GSE_datesets_results/finished_R_DEG/GSE174409_result/GSE174409_raw_counts_02102020.csv.gz", header = T, sep = '\t', data.table = F)
file_path <- "E:/drug_repurposing/GSE_datesets_results/finished_R_DEG/GSE174409_result/GSE174409_raw_counts_02102020.csv"
exprSet <- read.csv(file_path, header = TRUE, sep = ",", stringsAsFactors = FALSE)

fdata = fData(eSet[[1]]) 
pdata = pData(eSet[[1]]) 


gtf_data <- import("E:/drug_repurposing/GSE_datesets_results/finished_R_DEG/GSE174409_result/Homo_sapiens.GRCh38.113.gtf.gz")


gene_info <- data.frame(
  ensembl_gene_id = gtf_data$gene_id,
  gene_symbol = gtf_data$gene_name
)


gene_info <- unique(gene_info)


colnames(gene_info) <- c("X", "HUGOSYMBOL")


exprSet <- merge(exprSet, gene_info, by = "X", all.x = TRUE) 


exprSet$HUGOSYMBOL[is.na(exprSet$HUGOSYMBOL)] <- ""

last_col_name <- names(exprSet)[ncol(exprSet)]


exprSet <- exprSet[, c(last_col_name, setdiff(names(exprSet), last_col_name))]

exprSet <- exprSet[,-which(names(exprSet) == "X")]

exprSet <- exprSet[!is.na(exprSet$HUGOSYMBOL) & exprSet$HUGOSYMBOL != "", ]
names(exprSet)[1] <- "X"
exprSet_1 <- exprSet[!duplicated(exprSet$X), ] 
rownames(exprSet_1) <- exprSet_1$X  
exprSet_2 <- exprSet_1[-which(names(exprSet_1) == "X")]  

colnames(exprSet_2)[0:80] <- pdata$geo_accession  
dim(exprSet_2)  
dim(fdata) 
dim(pdata) 



opioid_exprSet <- c()
control_exprSet <- c()


for (i in 1:nrow(pdata)) {
  
  if (grepl('OUD', pdata$characteristics_ch1[i])) {
    
    opioid_exprSet <- c(opioid_exprSet, rownames(pdata)[i])
  } else {
    
    control_exprSet <- c(control_exprSet, rownames(pdata)[i])
  }
}

opioid_exprSet_data <- exprSet_2[, colnames(exprSet_2) %in% opioid_exprSet]
control_exprSet_data <- exprSet_2[, colnames(exprSet_2) %in% control_exprSet]
exprSet_final <- cbind(opioid_exprSet_data, control_exprSet_data) 

group <- c(rep('opioid', ncol(opioid_exprSet_data)), rep('control', ncol(control_exprSet_data)))

group <- factor(group, levels = c("control", "opioid"))    


colData <- data.frame(row.names = colnames(exprSet_final),
                      group = group)
colData$group <- factor(colData$group, levels = c("control", "opioid"))  

dds <- DESeqDataSetFromMatrix(countData = exprSet_final, 
                              colData = colData,        
                              design = ~ group)         

head(dds)  


dds <- DESeq(dds)


resultsNames(dds)

res <- results(dds, contrast = c("group", rev(levels(group))))


resOrdered <- res[order(res$padj), ]


DEG <- as.data.frame(resOrdered)


DEG_result <- na.omit(DEG)


logFC = 1
P.Value = 0.01


k1 <- (DEG_result$pvalue < P.Value) & (DEG_result$log2FoldChange < -logFC) 


k2 <- (DEG_result$pvalue < P.Value) & (DEG_result$log2FoldChange > logFC)


DEG_result <- mutate(DEG_result, change = ifelse(k1, "down", ifelse(k2, "up", "stable")))


filtered_DEG_result <- DEG_result %>%
  filter(!grepl("stable", change))
filtered_DEG_result$geneName <- rownames(filtered_DEG_result)


write.csv(filtered_DEG_result, file = './DESeq2_result/DEGs_DESeq2.csv', row.names = FALSE, fileEncoding = "UTF-8")


#table(DEG_result$change) 
# down stable     up 
#  693  43491   1339 


p <- ggplot(data = DEG_result, 
            aes(x = log2FoldChange, 
                y = -log10(pvalue))) +
  geom_point(alpha = 0.4, size = 3.5, 
             aes(color = change)) +
  ylab("-log10(Pvalue)")+
  scale_color_manual(values = c("blue4", "grey", "red3"))+
  geom_vline(xintercept = c(-logFC, logFC), lty = 4, col = "black", lwd = 0.8) +
  geom_hline(yintercept = -log10(P.Value), lty = 4, col = "black", lwd = 0.8) +
  theme_bw()
p 

ggsave(filename = "./volcano_plot_deseq2.pdf", plot = p, device = "pdf", width = 6, height = 5)
ggsave(filename = "./volcano_plot_deseq2.png", plot = p, device = "pdf", width = 6, height = 5)
dev.off() 

