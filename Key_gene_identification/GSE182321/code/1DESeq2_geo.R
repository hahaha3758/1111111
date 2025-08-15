getwd()
setwd("E:/drug_repurposing/GSE_datesets_results/finished_R_DEG/GSE182321_result/DESeq2_rersult/top300")

library(data.table)  
library(dplyr)       
library(ggplot2)     

library(DESeq2)      
library(GEOquery)
library(biomaRt)
library(AnnotationDbi)
library(org.Hs.eg.db)
library(clusterProfiler)

eSet <- getGEO("GSE182321", destdir = ".", getGPL = T) 
file_path <- "E:/drug_repurposing/GSE_datesets_results/finished_R_DEG/GSE182321_result/GSE182321_Raw_counts_matrix_UTHBC.csv"

exprSet = exprs(eSet[[1]]) 
fdata = fData(eSet[[1]]) 
pdata = pData(eSet[[1]]) 

exprSet <- read.csv(file_path, header = TRUE, sep = ",", stringsAsFactors = FALSE)


exprSet <- filter(exprSet, type == "protein_coding")
exprSet_2 <- exprSet

exprSet_2 <- distinct(exprSet_2, geneName, .keep_all = T)
rownames(exprSet_2) <- exprSet_2$geneName  
exprSet_2 <- exprSet_2[,-(1:3)]  

colnames(exprSet_2)[0:41] <- pdata$geo_accession  
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

logFC = 0.8
P.Value = 0.05


k1 <- (DEG_result$pvalue < P.Value) & (DEG_result$log2FoldChange < -logFC) 

k2 <- (DEG_result$pvalue < P.Value) & (DEG_result$log2FoldChange > logFC)


DEG_result <- mutate(DEG_result, change = ifelse(k1, "down", ifelse(k2, "up", "stable")))


filtered_DEG_result <- DEG_result %>%
  filter(!grepl("stable", change))
filtered_DEG_result$geneName <- rownames(filtered_DEG_result)


write.csv(filtered_DEG_result, file = './DEGs_DESeq2.csv', row.names = FALSE, fileEncoding = "UTF-8")


p <- ggplot(data = DEG_result, 
            aes(x = log2FoldChange, 
                y = -log10(pvalue))) +
  geom_point(alpha = 4, size = 3.5, 
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
