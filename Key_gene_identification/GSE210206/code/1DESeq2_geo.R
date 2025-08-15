
getwd()
setwd("E:/drug_repurposing/GSE_datesets_results/finished_R_DEG/GSE210206_result")

library(data.table)  
library(dplyr)       
library(ggplot2)     

library(DESeq2)      
library(GEOquery)
library(biomaRt)
library(AnnotationDbi)
library(org.Hs.eg.db)
library(clusterProfiler)


eSet <- getGEO("GSE210206", destdir = ".", getGPL = T) 
file_path <- "./GSE210206_counts.csv"

exprSet = exprs(eSet[[1]]) 
fdata = fData(eSet[[1]]) 
pdata = pData(eSet[[1]]) 

exprSet <- read.csv(file_path, header = TRUE, sep = ",", stringsAsFactors = FALSE)



httr::set_config(httr::config(timeout = 120))


ensembl <- useMart("ensembl", dataset = "hsapiens_gene_ensembl", host = "https://www.ensembl.org")

unique_gene_names <- exprSet$gene_symbol

hugo_symbols <- getBM(attributes = c("external_gene_name", "hgnc_symbol"),
                      filters = "external_gene_name",
                      values = unique_gene_names,
                      mart = ensembl)


colnames(hugo_symbols) <- c("gene_symbol", "HUGOSYMBOL")


exprSet <- merge(exprSet, hugo_symbols, by = "gene_symbol", all.x = TRUE)


exprSet$HUGOSYMBOL[is.na(exprSet$HUGOSYMBOL)] <- ""


last_col_name <- names(exprSet)[ncol(exprSet)]


exprSet <- exprSet[, c(last_col_name, setdiff(names(exprSet), last_col_name))]

exprSet <- exprSet[,-which(names(exprSet) == "gene_symbol")]

exprSet <- exprSet[!is.na(exprSet$HUGOSYMBOL) & exprSet$HUGOSYMBOL != "", ]

names(exprSet)[1] <- "gene_symbol"

colnames(exprSet)[2:19] <- pdata$geo_accession
dim(exprSet)  
dim(fdata) 
dim(pdata) 
exprSet_2 <- distinct(exprSet, gene_symbol, .keep_all = T) 
rownames(exprSet_2) <- exprSet_2$gene_symbol 
exprSet_2 <- exprSet_2[ , -1]  


opioid_exprSet <- c()
control_exprSet <- c()

for (i in 1:nrow(pdata)) { 
  element <- pdata$characteristics_ch1.2[i] 
  if (grepl(' PBS', element)) { 
    control_exprSet <- c(control_exprSet, rownames(pdata)[i])
  } else { 
    opioid_exprSet <- c(opioid_exprSet, rownames(pdata)[i])
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


save(DEG_result, file = './DEG_result.Rdata')

logFC = 1
P.Value = 0.01


k1 <- (DEG_result$pvalue < P.Value) & (DEG_result$log2FoldChange < -logFC) 


k2 <- (DEG_result$pvalue < P.Value) & (DEG_result$log2FoldChange > logFC)


DEG_result <- mutate(DEG_result, change = ifelse(k1, "down", ifelse(k2, "up", "stable")))


filtered_DEG_result <- DEG_result %>%
  filter(!grepl("stable", change))
filtered_DEG_result$geneName <- rownames(filtered_DEG_result)

write.csv(filtered_DEG_result, file = './DESeq2_result/DEGs_DESeq2.csv', row.names = FALSE, fileEncoding = "UTF-8")

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

