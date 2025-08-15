
library(Seurat)
library(dplyr)
library(Matrix)


setwd("E:/drug_repurposing/GSE_datesets_results/finished_R_DEG/GSE260711")


sample_dirs <- list.dirs(path = "GSE260711_RAW", recursive = FALSE, full.names = TRUE)


seurat_list <- lapply(sample_dirs, function(dir) {
  counts <- Read10X(data.dir = dir)
  sample_name <- basename(dir)
  
  

  
  group <- ifelse(sample_name %in% c("GSM8122883_KH001", "GSM8122885_KH003"), "control", "opioid")
  
  
  seurat_obj <- CreateSeuratObject(counts = counts, project = sample_name)
  seurat_obj@meta.data$group <- group
  return(seurat_obj)
})


merged_seurat <- merge(seurat_list[[1]], y = seurat_list[-1]) 


merged_seurat <- NormalizeData(merged_seurat)  


merged_seurat <- FindVariableFeatures(merged_seurat, selection.method = "vst", nfeatures = 2000)  



merged_seurat <- ScaleData(merged_seurat)       

merged_seurat <- RunPCA(merged_seurat, npcs = 50)  


merged_seurat <- FindNeighbors(merged_seurat, dims = 1:30)
merged_seurat <- FindClusters(merged_seurat, resolution = 0.5) 


merged_seurat <- RunUMAP(merged_seurat, dims = 1:30)


merged_seurat <- JoinLayers(merged_seurat, layers = c("counts", "data", "percent.mt"))


Idents(merged_seurat) <- "group"


markers <- FindMarkers(
  merged_seurat,
  ident.1 = "opioid",  
  ident.2 = "control",    
  logfc.threshold = 0.25, 
  
  min.pct = 0.25,          
  test.use = "wilcox"     
)


significant_markers <- markers %>%
  
  filter(p_val_adj < 0.01) %>%  
  arrange(desc(avg_log2FC)) %>%  
  mutate(direction = ifelse(avg_log2FC > 0, "Upregulated",
                            ifelse(avg_log2FC < 0, "Downregulated", "Normal")))



head(significant_markers, 10)



library(ggrepel)

significant_markers$geneName <- rownames(significant_markers)

volcano_plot <- ggplot(significant_markers, aes(x = avg_log2FC, y = -log10(p_val_adj))) +
  geom_point(aes(color = ifelse(avg_log2FC > 0, "Up", ifelse(avg_log2FC < 0, "Down", "Other")))) +
  geom_text_repel(data = subset(significant_markers, abs(avg_log2FC) > 1),
                  aes(label = geneName), max.overlaps = 20) +
  scale_color_manual(values = c("Up" = "blue", "Down" = "red", "Other" = "gray")) +
  theme_classic() +
  labs(x = "Log2 Fold Change", y = "-Log10(Adjusted p-value)", color = "Regulation")

print(volcano_plot)
ggsave("./volcano_plot.png", plot = last_plot(), dpi = 600)




DoHeatmap(merged_seurat, features = rownames(significant_markers)[1:10], group.by = "group")
ggsave("./DoHeatmap.png", plot = last_plot(), dpi = 600)


write.csv(significant_markers, file = "./DEGs_seurat.csv")




