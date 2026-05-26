param(
    [string]$DatasetDir = "",
    [string]$OutputDir = "",
    [string]$ApiUrl = "http://localhost:8000/api/agent/unet/analyze",
    [double]$Threshold = -1.0,
    [int]$Limit = 0,
    [int]$Top = 12
)

$ErrorActionPreference = "Stop"

$ProjectRoot = $PSScriptRoot
if (-not $DatasetDir) {
    $DatasetDir = Join-Path $ProjectRoot "Dataset\test"
}
if (-not $OutputDir) {
    $OutputDir = Join-Path $ProjectRoot "original_unet_project\runs\unet_agent_current\test_visual_qa"
}

$imagesDir = Join-Path $DatasetDir "images"
$labelsDir = Join-Path $DatasetDir "labels"
$masksDir = Join-Path $OutputDir "model_masks"
$previewsDir = Join-Path $OutputDir "previews"
$sheetsDir = Join-Path $OutputDir "sheets"

New-Item -ItemType Directory -Force -Path $OutputDir, $masksDir, $previewsDir, $sheetsDir | Out-Null

$progressPath = Join-Path $OutputDir "progress.json"
$csvPath = Join-Path $OutputDir "metrics.csv"
$summaryPath = Join-Path $OutputDir "summary.json"
$reportPath = Join-Path $OutputDir "report.md"

Add-Type -AssemblyName System.Drawing

Add-Type -ReferencedAssemblies "System.Drawing" -TypeDefinition @"
using System;
using System.IO;
using System.Linq;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;

public class SegMetrics
{
    public long Total;
    public long LabelPixels;
    public long PredPixels;
    public long TP;
    public long FP;
    public long FN;
    public long TN;
    public double Dice;
    public double IoU;
    public double Precision;
    public double Recall;
    public double FpRatio;
    public double FnRatio;
}

public static class SegQaTools
{
    private static byte[] LoadGray(string path, int width, int height, bool nearest)
    {
        Image src = Image.FromFile(path);
        Bitmap bmp = new Bitmap(width, height, PixelFormat.Format32bppArgb);
        Graphics gfx = Graphics.FromImage(bmp);
        gfx.Clear(Color.Black);
        gfx.InterpolationMode = nearest ? InterpolationMode.NearestNeighbor : InterpolationMode.HighQualityBilinear;
        gfx.PixelOffsetMode = PixelOffsetMode.Half;
        gfx.DrawImage(src, 0, 0, width, height);
        gfx.Dispose();
        src.Dispose();

        Rectangle rect = new Rectangle(0, 0, width, height);
        BitmapData data = bmp.LockBits(rect, ImageLockMode.ReadOnly, PixelFormat.Format32bppArgb);
        int stride = data.Stride;
        byte[] raw = new byte[stride * height];
        Marshal.Copy(data.Scan0, raw, 0, raw.Length);
        bmp.UnlockBits(data);
        bmp.Dispose();

        byte[] gray = new byte[width * height];
        for (int y = 0; y < height; y++)
        {
            int row = y * stride;
            int dst = y * width;
            for (int x = 0; x < width; x++)
            {
                int i = row + x * 4;
                int b = raw[i + 0];
                int g = raw[i + 1];
                int r = raw[i + 2];
                gray[dst + x] = (byte)((r * 30 + g * 59 + b * 11) / 100);
            }
        }
        return gray;
    }

    public static SegMetrics EvaluateMask(string labelPath, string predPath)
    {
        Image labelImage = Image.FromFile(labelPath);
        int width = labelImage.Width;
        int height = labelImage.Height;
        labelImage.Dispose();

        byte[] label = LoadGray(labelPath, width, height, true);
        byte[] pred = LoadGray(predPath, width, height, true);
        SegMetrics m = new SegMetrics();
        m.Total = (long)width * (long)height;

        for (int i = 0; i < label.Length; i++)
        {
            bool l = label[i] > 127;
            bool p = pred[i] > 127;
            if (l) m.LabelPixels++;
            if (p) m.PredPixels++;
            if (l && p) m.TP++;
            else if (!l && p) m.FP++;
            else if (l && !p) m.FN++;
            else m.TN++;
        }

        long diceDen = 2 * m.TP + m.FP + m.FN;
        long iouDen = m.TP + m.FP + m.FN;
        m.Dice = diceDen == 0 ? 1.0 : (2.0 * m.TP) / diceDen;
        m.IoU = iouDen == 0 ? 1.0 : (1.0 * m.TP) / iouDen;
        m.Precision = (m.TP + m.FP) == 0 ? 1.0 : (1.0 * m.TP) / (m.TP + m.FP);
        m.Recall = (m.TP + m.FN) == 0 ? 1.0 : (1.0 * m.TP) / (m.TP + m.FN);
        m.FpRatio = m.Total == 0 ? 0.0 : (1.0 * m.FP) / m.Total;
        m.FnRatio = m.Total == 0 ? 0.0 : (1.0 * m.FN) / m.Total;
        return m;
    }

    private static Bitmap BuildPanel(byte[] gray, byte[] label, byte[] pred, int width, int height, string mode)
    {
        Bitmap bmp = new Bitmap(width, height, PixelFormat.Format32bppArgb);
        Rectangle rect = new Rectangle(0, 0, width, height);
        BitmapData data = bmp.LockBits(rect, ImageLockMode.WriteOnly, PixelFormat.Format32bppArgb);
        int stride = data.Stride;
        byte[] raw = new byte[stride * height];

        for (int y = 0; y < height; y++)
        {
            int row = y * stride;
            int srcRow = y * width;
            for (int x = 0; x < width; x++)
            {
                int idx = srcRow + x;
                int dst = row + x * 4;
                int baseGray = gray[idx];
                int r = baseGray;
                int g = baseGray;
                int b = baseGray;
                bool l = label != null && label[idx] > 127;
                bool p = pred != null && pred[idx] > 127;

                if (mode == "label" && l)
                {
                    r = (int)(baseGray * 0.25);
                    g = Math.Min(255, (int)(baseGray * 0.35) + 170);
                    b = (int)(baseGray * 0.25);
                }
                else if (mode == "pred" && p)
                {
                    r = Math.Min(255, (int)(baseGray * 0.35) + 180);
                    g = (int)(baseGray * 0.25);
                    b = (int)(baseGray * 0.25);
                }
                else if (mode == "error")
                {
                    if (l && p)
                    {
                        r = 255; g = 210; b = 35;
                    }
                    else if (!l && p)
                    {
                        r = 255; g = 40; b = 40;
                    }
                    else if (l && !p)
                    {
                        r = 30; g = 210; b = 255;
                    }
                }

                raw[dst + 0] = (byte)b;
                raw[dst + 1] = (byte)g;
                raw[dst + 2] = (byte)r;
                raw[dst + 3] = 255;
            }
        }

        Marshal.Copy(raw, 0, data.Scan0, raw.Length);
        bmp.UnlockBits(data);
        return bmp;
    }

    public static void CreatePreview(string imagePath, string labelPath, string predPath, string outputPath, string title)
    {
        int panel = 256;
        int gap = 16;
        int header = 58;
        int footer = 34;
        int width = panel * 4 + gap * 5;
        int height = header + panel + footer;

        byte[] gray = LoadGray(imagePath, panel, panel, false);
        byte[] label = LoadGray(labelPath, panel, panel, true);
        byte[] pred = LoadGray(predPath, panel, panel, true);

        Bitmap canvas = new Bitmap(width, height, PixelFormat.Format32bppArgb);
        Graphics gfx = Graphics.FromImage(canvas);
        gfx.SmoothingMode = SmoothingMode.AntiAlias;
        gfx.Clear(Color.FromArgb(12, 18, 34));

        Font titleFont = new Font(FontFamily.GenericSansSerif, 13, FontStyle.Bold);
        Font labelFont = new Font(FontFamily.GenericSansSerif, 10, FontStyle.Bold);
        Brush white = new SolidBrush(Color.White);
        Brush muted = new SolidBrush(Color.FromArgb(176, 194, 224));
        Pen framePen = new Pen(Color.FromArgb(55, 76, 120), 1);
        gfx.DrawString(title, titleFont, white, 16, 14);

        string[] names = new string[] { "Original", "Label", "Model prediction", "Error map" };
        Bitmap[] panels = new Bitmap[] {
            BuildPanel(gray, null, null, panel, panel, "original"),
            BuildPanel(gray, label, null, panel, panel, "label"),
            BuildPanel(gray, null, pred, panel, panel, "pred"),
            BuildPanel(gray, label, pred, panel, panel, "error")
        };

        for (int i = 0; i < 4; i++)
        {
            int x = gap + i * (panel + gap);
            gfx.DrawImage(panels[i], x, header, panel, panel);
            gfx.DrawRectangle(framePen, x, header, panel, panel);
            gfx.DrawString(names[i], labelFont, muted, x + 4, header + panel + 7);
            panels[i].Dispose();
        }

        gfx.DrawString("error map: yellow=TP, red=FP, cyan=FN", labelFont, muted, width - 330, header + panel + 7);
        titleFont.Dispose();
        labelFont.Dispose();
        white.Dispose();
        muted.Dispose();
        framePen.Dispose();
        gfx.Dispose();

        Directory.CreateDirectory(Path.GetDirectoryName(outputPath));
        canvas.Save(outputPath, ImageFormat.Png);
        canvas.Dispose();
    }

    public static void CreateContactSheet(string[] previewPaths, string outputPath, int columns)
    {
        if (previewPaths == null || previewPaths.Length == 0) return;
        if (columns <= 0) columns = 2;

        Bitmap first = new Bitmap(previewPaths[0]);
        int itemW = first.Width;
        int itemH = first.Height;
        first.Dispose();

        int rows = (int)Math.Ceiling(previewPaths.Length / (double)columns);
        int width = itemW * columns;
        int height = itemH * rows;
        Bitmap sheet = new Bitmap(width, height, PixelFormat.Format32bppArgb);
        Graphics gfx = Graphics.FromImage(sheet);
        gfx.Clear(Color.FromArgb(8, 12, 24));
        gfx.InterpolationMode = InterpolationMode.HighQualityBicubic;

        for (int i = 0; i < previewPaths.Length; i++)
        {
            Bitmap item = new Bitmap(previewPaths[i]);
            int col = i % columns;
            int row = i / columns;
            gfx.DrawImage(item, col * itemW, row * itemH, itemW, itemH);
            item.Dispose();
        }

        Directory.CreateDirectory(Path.GetDirectoryName(outputPath));
        sheet.Save(outputPath, ImageFormat.Png);
        gfx.Dispose();
        sheet.Dispose();
    }
}
"@

function Write-ProgressFile {
    param(
        [int]$Done,
        [int]$Total,
        [string]$Current,
        [double]$WorstDice,
        [string]$State = "running"
    )

    [PSCustomObject]@{
        state = $State
        done = $Done
        total = $Total
        current = $Current
        worst_dice = [math]::Round($WorstDice, 6)
        updated_at = (Get-Date).ToString("s")
    } | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $progressPath -Encoding UTF8
}

function Save-MaskFromApi {
    param(
        [string]$ImagePath,
        [string]$MaskPath
    )

    $curlArgs = @("-s", "-X", "POST", "-F", "file=@$ImagePath")
    if ($Threshold -ge 0.0) {
        $thresholdText = $Threshold.ToString([Globalization.CultureInfo]::InvariantCulture)
        $curlArgs += @("-F", "threshold=$thresholdText")
    }
    $curlArgs += $ApiUrl
    $responseText = & curl.exe @curlArgs
    if ($LASTEXITCODE -ne 0) {
        throw "curl failed for $ImagePath"
    }
    $response = $responseText | ConvertFrom-Json
    if (-not $response.success) {
        throw "U-Net API returned success=false for $ImagePath"
    }
    $maskB64 = [string]$response.segmentation_mask
    if ([string]::IsNullOrWhiteSpace($maskB64)) {
        throw "U-Net API returned an empty mask for $ImagePath"
    }
    [IO.File]::WriteAllBytes($MaskPath, [Convert]::FromBase64String($maskB64))
    return $response
}

$images = Get-ChildItem -LiteralPath $imagesDir -Filter *.tif | Sort-Object Name
if ($Limit -gt 0) {
    $images = $images | Select-Object -First $Limit
}

$total = @($images).Count
if ($total -eq 0) {
    throw "No test images found in $imagesDir"
}

$rows = New-Object System.Collections.Generic.List[object]
$failures = New-Object System.Collections.Generic.List[object]
$worstDice = 1.0
Write-ProgressFile -Done 0 -Total $total -Current "" -WorstDice $worstDice

$index = 0
foreach ($image in $images) {
    $index += 1
    $sample = [IO.Path]::GetFileNameWithoutExtension($image.Name)
    $labelPath = Join-Path $labelsDir ($sample + ".tif")
    $maskPath = Join-Path $masksDir ($sample + "_mask.png")

    try {
        if (-not (Test-Path -LiteralPath $labelPath)) {
            throw "Missing label: $labelPath"
        }

        Save-MaskFromApi -ImagePath $image.FullName -MaskPath $maskPath | Out-Null
        $m = [SegQaTools]::EvaluateMask($labelPath, $maskPath)
        $worstDice = [Math]::Min($worstDice, [double]$m.Dice)

        $rows.Add([PSCustomObject]@{
            sample = $sample
            image = $image.FullName
            label = $labelPath
            mask = $maskPath
            dice = [math]::Round([double]$m.Dice, 8)
            iou = [math]::Round([double]$m.IoU, 8)
            precision = [math]::Round([double]$m.Precision, 8)
            recall = [math]::Round([double]$m.Recall, 8)
            label_pixels = $m.LabelPixels
            pred_pixels = $m.PredPixels
            tp = $m.TP
            fp = $m.FP
            fn = $m.FN
            fp_ratio = [math]::Round([double]$m.FpRatio, 8)
            fn_ratio = [math]::Round([double]$m.FnRatio, 8)
        }) | Out-Null
    }
    catch {
        $failures.Add([PSCustomObject]@{
            sample = $sample
            image = $image.FullName
            error = $_.Exception.Message
        }) | Out-Null
    }

    Write-ProgressFile -Done $index -Total $total -Current $sample -WorstDice $worstDice
}

[object[]]$rowsArray = $rows.ToArray()
[object[]]$failuresArray = $failures.ToArray()
$rowsArray | Sort-Object dice | Export-Csv -LiteralPath $csvPath -NoTypeInformation -Encoding UTF8

if ($rowsArray.Count -gt 0) {
    $meanDice = ($rowsArray | Measure-Object -Property dice -Average).Average
    $meanIoU = ($rowsArray | Measure-Object -Property iou -Average).Average
    $minDice = ($rowsArray | Measure-Object -Property dice -Minimum).Minimum
    $maxDice = ($rowsArray | Measure-Object -Property dice -Maximum).Maximum
    $emptyLabels = @($rowsArray | Where-Object { $_.label_pixels -eq 0 }).Count
    $emptyPreds = @($rowsArray | Where-Object { $_.pred_pixels -eq 0 }).Count

    $worst = @($rowsArray | Sort-Object dice | Select-Object -First $Top)
    $best = @($rowsArray | Sort-Object dice -Descending | Select-Object -First ([Math]::Min(6, $rowsArray.Count)))
    $mostFp = @($rowsArray | Sort-Object fp -Descending | Select-Object -First ([Math]::Min(6, $rowsArray.Count)))
    $mostFn = @($rowsArray | Sort-Object fn -Descending | Select-Object -First ([Math]::Min(6, $rowsArray.Count)))

    function New-Previews {
        param(
            [object[]]$Items,
            [string]$Group
        )

        $paths = New-Object System.Collections.Generic.List[string]
        foreach ($item in $Items) {
            $previewPath = Join-Path $previewsDir ($Group + "_" + $item.sample + ".png")
            $title = ("{0} | Dice={1:N4} IoU={2:N4} P={3:N4} R={4:N4} FP={5} FN={6}" -f $item.sample, [double]$item.dice, [double]$item.iou, [double]$item.precision, [double]$item.recall, $item.fp, $item.fn)
            [SegQaTools]::CreatePreview($item.image, $item.label, $item.mask, $previewPath, $title)
            $paths.Add($previewPath) | Out-Null
        }
        return @($paths)
    }

    $worstPreviewPaths = New-Previews -Items $worst -Group "worst"
    $bestPreviewPaths = New-Previews -Items $best -Group "best"
    $fpPreviewPaths = New-Previews -Items $mostFp -Group "fp"
    $fnPreviewPaths = New-Previews -Items $mostFn -Group "fn"

    [SegQaTools]::CreateContactSheet($worstPreviewPaths, (Join-Path $sheetsDir "worst_cases.png"), 2)
    [SegQaTools]::CreateContactSheet($bestPreviewPaths, (Join-Path $sheetsDir "best_cases.png"), 2)
    [SegQaTools]::CreateContactSheet($fpPreviewPaths, (Join-Path $sheetsDir "largest_false_positive.png"), 2)
    [SegQaTools]::CreateContactSheet($fnPreviewPaths, (Join-Path $sheetsDir "largest_false_negative.png"), 2)

    $summary = [PSCustomObject]@{
        dataset_dir = $DatasetDir
        output_dir = $OutputDir
        api_url = $ApiUrl
        threshold = if ($Threshold -ge 0.0) { $Threshold } else { "agent_default" }
        evaluated_samples = $rowsArray.Count
        failures = $failuresArray.Count
        mean_dice = [math]::Round([double]$meanDice, 8)
        mean_iou = [math]::Round([double]$meanIoU, 8)
        min_dice = [math]::Round([double]$minDice, 8)
        max_dice = [math]::Round([double]$maxDice, 8)
        empty_labels = $emptyLabels
        empty_predictions = $emptyPreds
        worst_cases = @($worst | Select-Object sample, dice, iou, precision, recall, fp, fn, label_pixels, pred_pixels)
        best_cases = @($best | Select-Object sample, dice, iou, precision, recall, fp, fn, label_pixels, pred_pixels)
        largest_false_positive = @($mostFp | Select-Object sample, dice, iou, precision, recall, fp, fn, label_pixels, pred_pixels)
        largest_false_negative = @($mostFn | Select-Object sample, dice, iou, precision, recall, fp, fn, label_pixels, pred_pixels)
        sheets = [PSCustomObject]@{
            worst_cases = Join-Path $sheetsDir "worst_cases.png"
            best_cases = Join-Path $sheetsDir "best_cases.png"
            largest_false_positive = Join-Path $sheetsDir "largest_false_positive.png"
            largest_false_negative = Join-Path $sheetsDir "largest_false_negative.png"
        }
        generated_at = (Get-Date).ToString("s")
    }
    $summary | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $summaryPath -Encoding UTF8

    $report = @()
    $report += "# Test Set Visual QA"
    $report += ""
    $report += "- evaluated samples: $($rowsArray.Count)"
    $report += "- failures: $($failuresArray.Count)"
    $report += "- mean Dice: $([math]::Round([double]$meanDice, 6))"
    $report += "- mean IoU: $([math]::Round([double]$meanIoU, 6))"
    $report += "- min Dice: $([math]::Round([double]$minDice, 6))"
    $report += "- max Dice: $([math]::Round([double]$maxDice, 6))"
    $report += "- empty labels: $emptyLabels"
    $report += "- empty predictions: $emptyPreds"
    $report += ""
    $report += "## Sheets"
    $report += ""
    $report += "- worst cases: $((Join-Path $sheetsDir 'worst_cases.png'))"
    $report += "- best cases: $((Join-Path $sheetsDir 'best_cases.png'))"
    $report += "- largest false positive: $((Join-Path $sheetsDir 'largest_false_positive.png'))"
    $report += "- largest false negative: $((Join-Path $sheetsDir 'largest_false_negative.png'))"
    $report += ""
    $report += "## Worst cases"
    foreach ($item in $worst) {
        $report += ("- {0}: Dice={1:N4}, IoU={2:N4}, FP={3}, FN={4}, label={5}, pred={6}" -f $item.sample, [double]$item.dice, [double]$item.iou, $item.fp, $item.fn, $item.label_pixels, $item.pred_pixels)
    }
    $report | Set-Content -LiteralPath $reportPath -Encoding UTF8
}

if ($failuresArray.Count -gt 0) {
    $failuresArray | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $OutputDir "failures.json") -Encoding UTF8
}

Write-ProgressFile -Done $total -Total $total -Current "" -WorstDice $worstDice -State "done"
Write-Host "QA complete: $OutputDir"
