# # Ensure BiocManager is installed
# if(!requireNamespace("BiocManager", quietly = TRUE))
#   install.packages("BiocManager")
getwd()
setwd("E:/drug_repurposing/GSE_datesets_results/finished_R_DEG/GSE167922_result")
# Load required libraries
library(GEOquery) 
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
library(readr)


# Download data from GEO
eSet <- getGEO("GSE167922", destdir = ".", getGPL = T) 
file_path <- "./GSE167922_raw_counts_GRCh38.p13_NCBI.tsv"

exprSet = exprs(eSet[[1]]) 
fdata = fData(eSet[[1]]) 
pdata = pData(eSet[[1]]) 

exprSet <- read_tsv(file_path)
# exprSet <- read.csv(file_path, header = TRUE, sep = ",", stringsAsFactors = FALSE,col.names=pdata$geo_accession)


unique_gene_names <- exprSet$GeneID


file_path1 <- "./Human.GRCh38.p13.annot.tsv"
hugo_symbols <- read_tsv(file_path1)

selected_columns <- hugo_symbols[, 1:2]


exprSet <- merge(exprSet, selected_columns, by = "GeneID", all.x = TRUE)


exprSet$Symbol[is.na(exprSet$Symbol)] <- ""


last_col_name <- names(exprSet)[ncol(exprSet)]


exprSet <- exprSet[, c(last_col_name, setdiff(names(exprSet), last_col_name))]

exprSet <- exprSet[,-which(names(exprSet) == "GeneID")]

exprSet <- exprSet[!is.na(exprSet$Symbol) & exprSet$Symbol != "", ]

names(exprSet)[1] <- "X"


exprSet <- exprSet[!grepl("^LOC", exprSet$X), ]


# colnames(exprSet)[2:43] <- pdata$geo_accession
dim(exprSet)  
dim(fdata) 
dim(pdata) 
exprSet_2 <- distinct(exprSet, X, .keep_all = T) 
rownames(exprSet_2) <- exprSet_2$X 
exprSet_2 <- exprSet_2[ , -1] 

opioid_exprSet <- c()
control_exprSet <- c()



for (i in 1:nrow(pdata)) {
 
  if (grepl('fentanyl', pdata$title[i])) {
   
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
# [1] "Intercept"             "group_tumor_vs_normal"


res <- results(dds, contrast = c("group", rev(levels(group))))

# res <- results(dds, contrast = c("group", levels(group)[2], levels(group)[1]))

resOrdered <- res[order(res$padj), ]


DEG <- as.data.frame(resOrdered)


DEG_result <- na.omit(DEG)

# 将处理后的差异表达结果保存为R数据文件
#save(DEG_result, file = './DEG_result.Rdata')


#load("./data/DEG_deseq2.Rdata")
#load("./data/DEG_deseq2.Rdata")

logFC = 0.2   #1
P.Value = 0.1

k1 <- (DEG_result$pvalue < P.Value) & (DEG_result$log2FoldChange < -logFC) 


k2 <- (DEG_result$pvalue < P.Value) & (DEG_result$log2FoldChange > logFC)


DEG_result <- mutate(DEG_result, change = ifelse(k1, "down", ifelse(k2, "up", "stable")))


filtered_DEG_result <- DEG_result %>%
  filter(!grepl("stable", change))
filtered_DEG_result$geneName <- rownames(filtered_DEG_result)


write.csv(filtered_DEG_result, file = './DESeq2_result/top300/DEGs_DESeq2.csv', row.names = TRUE, fileEncoding = "UTF-8")


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
ggsave(filename = "./DESeq2_result/top300/volcano_plot_deseq2.pdf", plot = p, device = "pdf", width = 6, height = 5)
ggsave(filename = "./DESeq2_result/top300/volcano_plot_deseq2.png", plot = p, device = "pdf", width = 6, height = 5)


