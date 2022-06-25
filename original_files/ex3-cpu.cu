///////////////////////////////////////////////// DO NOT CHANGE ///////////////////////////////////////

#include "ex3.h"

/**
 * Perform interpolation
 *
 * @param maps 3d array ([TILES_COUNT][TILES_COUNT][256]) of the tiles’ maps.
 * @param tile_count the number of tiles fits in image width.
 * @param maps col the column of the pixel in the image. 
 * @param maps row the row of the pixel in the image.
 * @param pixel_value the value of the pixel (col, row).
 * @return the new value of the pixel (col, row).
 */
uchar interpolate(uchar* maps, int tile_count, int col, int row, uchar pixel_val){
            int including_tile_x = col/TILE_WIDTH;
            int including_tile_y = row/TILE_WIDTH;

            int x_in_tile_offset = col%TILE_WIDTH;
            int y_in_tile_offset = row%TILE_WIDTH;
            bool is_in_tile_left_half = x_in_tile_offset < (TILE_WIDTH/2);
            bool is_in_tile_top_half = y_in_tile_offset < (TILE_WIDTH/2);
            int left = is_in_tile_left_half ? including_tile_x - 1 : including_tile_x;
            int right = left + 1;
            int top = is_in_tile_top_half ? including_tile_y - 1 : including_tile_y;
            int bottom = top + 1;

            left = left < 0 ? 0 : left;
            right = right >= tile_count ? tile_count-1 : right;
            top = top < 0 ? 0 : top;
            bottom = bottom >= tile_count ? tile_count-1 : bottom;


            uchar top_left      = *(maps + 256 * (top * tile_count + left) + pixel_val);        //maps[top][left][pixel_val]
            uchar top_right     = *(maps + 256 * (top * tile_count + right) + pixel_val);       //maps[top][right][pixel_val]
            uchar bottom_left   = *(maps + 256 * (bottom * tile_count + left) + pixel_val);     //maps[bottom][left][pixel_val]
            uchar bottom_right  = *(maps + 256 * (bottom * tile_count + right) + pixel_val);    //maps[bottom][right][pixel_val]

            int alpha_int = x_in_tile_offset + (is_in_tile_left_half ? 1 : -1)*(TILE_WIDTH/2);
            int beta_int = y_in_tile_offset + (is_in_tile_top_half ? 1 : -1)*(TILE_WIDTH/2);
            float alpha = (float)alpha_int / TILE_WIDTH;
            float beta = (float)beta_int / TILE_WIDTH;

            uchar new_val = (1-alpha)   * (1-beta) * top_left      + 
                            alpha       * (1-beta) * top_right     +
                            (1-alpha)   * beta     * bottom_left   +
                            alpha       * beta     * bottom_right;

            return new_val;
}


void cpu_process(uchar *img_in, uchar *img_out, int width, int height) {
    const int tile_size = TILE_WIDTH*TILE_WIDTH;
    int tile_count = width / TILE_WIDTH;
    uchar maps[tile_count][tile_count][256];

    for (int i=0; i<tile_count; i++){
        for (int j=0; j<tile_count; j++){
            int histogram[256] = { 0 };
            int left = TILE_WIDTH*j;
            int right = TILE_WIDTH*(j+1) - 1;
            int top = TILE_WIDTH*i;
            int bottom = TILE_WIDTH*(i+1) - 1;

            for (int y=top; y<=bottom; y++) {
                for (int x=left; x<=right; x++) {
                    uchar* row = img_in + y*width;
                    histogram[row[x]]++;
                }
            }
            
            int cdf[256] = { 0 };
            int hist_sum = 0;
            for (int k = 0; k < 256; k++) {
                hist_sum += histogram[k];
                cdf[k] = hist_sum;
            }

            uchar* map = maps[i][j];
            for (int k = 0; k < 256; k++) {
                map[k] = (float(cdf[k]) * 255) / (tile_size);
            }

        }
    }

    for (int row=0; row<height; row++) {
        for (int col=0; col<width; col++) {
            uchar pixel_val = img_in[row * width + col];
            uchar new_val;
            new_val = interpolate((uchar*)maps, tile_count ,col, row, pixel_val);
            img_out[row * width + col] = new_val;
        }
    }
}
