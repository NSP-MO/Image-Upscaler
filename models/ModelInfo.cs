namespace ImageUpscaler.Models
{
    public enum UpscalerType
    {
        FastLanczos,
        FastNedi,
        GuidedEdge,
        RaisrPatch,
        XbrzPattern,
        VectorContour,
        NeuralEsrgan,
        NeuralSwinir,
        NeuralDat
    }

    public class ModelInfo
    {
        public string Id { get; set; } = string.Empty;
        public string Name { get; set; } = string.Empty;
        public string Description { get; set; } = string.Empty;
        public UpscalerType Type { get; set; }
        public int DefaultScale { get; set; } = 4;
        public string? Filename { get; set; }
        public string? Url { get; set; }
        public bool IsDownloaded { get; set; } = true;

        public override string ToString() => Name;
    }
}
