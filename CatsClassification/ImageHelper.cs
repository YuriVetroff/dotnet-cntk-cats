using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.Linq;
using System.Threading.Tasks;

namespace CatsClassification
{
    public static class ImageHelper
    {
        private const int DEFAULT_HORIZONTAL_RESOLUTION = 96;
        private const int DEFAULT_VERTICAL_RESOLUTION = 96;

        public static float[] Load(int width, int height, string filePath)
        {
            var bitmap = new Bitmap(Image.FromFile(filePath));
            var resized = Resize(bitmap, width, height);
            var resizedCHW = ParallelExtractCHW(resized);
            return resizedCHW;
        }

        private static float[] ParallelExtractCHW(Bitmap image)
        {
            // We use local variables to avoid contention on the image object through the multiple threads.
            int channelStride = image.Width * image.Height;
            int imageWidth = image.Width;
            int imageHeight = image.Height;

            var features = new byte[imageWidth * imageHeight * 3];
            var bitmapData = image.LockBits(new System.Drawing.Rectangle(0, 0, imageWidth, imageHeight), ImageLockMode.ReadOnly, image.PixelFormat);
            IntPtr ptr = bitmapData.Scan0;
            int bytes = Math.Abs(bitmapData.Stride) * bitmapData.Height;
            byte[] rgbValues = new byte[bytes];

            int stride = bitmapData.Stride;

            // Copy the RGB values into the array.
            System.Runtime.InteropServices.Marshal.Copy(ptr, rgbValues, 0, bytes);

            // The mapping depends on the pixel format
            // The mapPixel lambda will return the right color channel for the desired pixel
            Func<int, int, int, int> mapPixel = GetPixelMapper(image.PixelFormat, stride);

            Parallel.For(0, imageHeight, (int h) =>
            {
                Parallel.For(0, imageWidth, (int w) =>
                {
                    Parallel.For(0, 3, (int c) =>
                    {
                        features[channelStride * c + imageWidth * h + w] = rgbValues[mapPixel(h, w, c)];
                    });
                });
            });

            image.UnlockBits(bitmapData);

            return features.Select(b => (float)b).ToArray();
        }
        private static Func<int, int, int, int> GetPixelMapper(PixelFormat pixelFormat, int heightStride)
        {
            switch (pixelFormat)
            {
                case PixelFormat.Format32bppArgb:
                    return (h, w, c) => h * heightStride + w * 4 + c;  // bytes are B-G-R-A
                case PixelFormat.Format24bppRgb:
                default:
                    return (h, w, c) => h * heightStride + w * 3 + c;  // bytes are B-G-R
            }
        }
        private static Bitmap Resize(Bitmap image, int width, int height)
        {
            var newImg = new Bitmap(width, height);

            newImg.SetResolution(
                image.HorizontalResolution == 0
                    ? DEFAULT_HORIZONTAL_RESOLUTION
                    : image.HorizontalResolution,
                image.VerticalResolution == 0
                    ? DEFAULT_VERTICAL_RESOLUTION
                    : image.VerticalResolution);

            using (var graphics = Graphics.FromImage(newImg))
            {
                graphics.CompositingMode = System.Drawing.Drawing2D.CompositingMode.SourceCopy;

                graphics.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.HighQualityBicubic;
                graphics.CompositingQuality = System.Drawing.Drawing2D.CompositingQuality.HighQuality;
                graphics.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.HighQuality;


                var attributes = new ImageAttributes();
                attributes.SetWrapMode(System.Drawing.Drawing2D.WrapMode.TileFlipXY);
                graphics.DrawImage(image, new System.Drawing.Rectangle(0, 0, width, height), 0, 0, image.Width, image.Height, GraphicsUnit.Pixel, attributes);
            }

            return newImg;
        }
    }
}
