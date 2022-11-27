use ndarray::{Array, Array2, Array4, arr2, stack, Axis, Zip, s};
use ndarray_parallel::prelude::*;

// 計算できない値についてはNaNを入れる

// 命名規則(必ず新しい規則はここに書く)
// 1->2->3の順に書いていく eg. u_hori_nxnx
// 1. f:粒子密度  feq:eq場の粒子密度  u:風速  rho:密度
// 1. w0,w1,w2,w3,w4:そのレイヤーの重み  dw0,dw1,dw2,dw3,dw4:重みの変化分
// 2. _vert:縦,緯線方向(下向き正！)  _hori:横,経線方向(右向き正)
// 3. _nx:次の  _nxnx:次の次の  _prev:前の
// row:行数  col:列数  r:r行(添字)  c:c列(添字) dr, dc
// C:係数

const C: [[f64; 3]; 3] = [[1.0/36.0, 1.0/9.0, 1.0/36.0], [1.0/9.0, 4.0/9.0, 1.0/9.0], [1.0/36.0, 1.0/9.0, 1.0/36.0]];
const ERROR_DELTA: f64 = 0.00000000001;

pub struct InputField {
    row: usize,
    col: usize,
    f: Array4<f64>,
    u_vert: Array2<f64>,
    u_hori: Array2<f64>,
    // u2: Array2<f64>,
    rho: Array2<f64>,
}

pub struct StreamingWeight {
    row: usize,
    col: usize,
    margin: usize,
    w0: Array4<f64>,
    w1: Array4<f64>,
    delta: Array4<f64>,
}

pub struct StreamedField {
    row: usize,
    col: usize,
    margin: usize,
    f: Array4<f64>,
    u_vert: Array2<f64>,
    u_hori: Array2<f64>,
    rho: Array2<f64>,
}

impl InputField {
    pub fn new(row: usize, col: usize) -> InputField {
        let f = Array4::<f64>::zeros((row, col, 3, 3));
        let u_vert = Array2::<f64>::zeros((row, col));
        let u_hori = Array2::<f64>::zeros((row, col));
        let rho = Array2::<f64>::zeros((row, col));
        InputField { row, col, f, u_vert, u_hori, rho }
    }

    pub fn set(self: &mut Self, u_vert: Array2<f64>, u_hori: Array2<f64>, rho: Array2<f64>) {
        if [self.row, self.col] != u_vert.shape() || [self.row, self.col] != u_hori.shape() || [self.row, self.col] != rho.shape() {
            panic!("panicked at line {} in {}", line!(), file!());
        }
        self.u_vert = u_vert;
        self.u_hori = u_hori;
        self.rho = rho;

        for dr in -1..=1_i32 { 
            for dc in -1..=1_i32 {
                let f_slice = self.f.slice_mut(s![.., .., dr+1, dc+1]);
                Zip::from(f_slice).and(&self.u_vert).and(&self.u_hori).and(&self.rho)
                    .for_each(|f, u_vert, u_hori, rho| {
                        let dr_f = dr as f64; // -1., 0., 1.のいずれかに変換
                        let dc_f = dc as f64;
                        let u2 = u_vert * u_vert + u_hori * u_hori; // ここちょっと無駄な気もする
                        let u_prod = u_vert * dr_f + u_hori * dc_f;
                        *f = C[(dr+1) as usize][(dc+1) as usize] * rho * (1.0 + (3.0 + 4.5 * u_prod) * u_prod - 1.5 * u2);
                    })
            }
        }
    }
}

impl StreamingWeight {
    pub fn new(row: usize, col: usize, margin: usize) -> StreamingWeight {
        let mut w0 = Array4::<f64>::from_elem((row, col, 3, 3), f64::NAN);
        let mut w1 = Array4::<f64>::from_elem((row, col, 3, 3), f64::NAN);
        let mut delta = Array4::<f64>::from_elem((row, col, 3, 3), f64::NAN);
        w0.slice_mut(s![margin..row-margin, margin..col-margin, .., ..]).fill(0.0);
        w1.slice_mut(s![margin..row-margin, margin..col-margin, .., ..]).fill(1.0);
        delta.slice_mut(s![margin..row-margin, margin..col-margin, .., ..]).fill(0.0);
        StreamingWeight { row, col, margin, w0, w1, delta }
    }
}

impl StreamedField {
    pub fn new(row: usize, col: usize, margin: usize) -> StreamedField {
        let mut f = Array4::<f64>::from_elem((row, col, 3, 3), f64::NAN);
        let mut u_vert = Array2::<f64>::from_elem((row, col), f64::NAN);
        let mut u_hori = Array2::<f64>::from_elem((row, col), f64::NAN);
        let mut rho = Array2::<f64>::from_elem((row, col), f64::NAN);
        f.slice_mut(s![margin..row-margin, margin..col-margin, .., ..]).fill(0.0);
        u_vert.slice_mut(s![margin..row-margin, margin..col-margin]).fill(0.0);
        u_hori.slice_mut(s![margin..row-margin, margin..col-margin]).fill(0.0);
        rho.slice_mut(s![margin..row-margin, margin..col-margin]).fill(0.0);
        StreamedField { row, col, margin, f, u_vert, u_hori, rho }
    }

    pub fn stream(self: &mut Self, input_field: &InputField, streaming_weight: &StreamingWeight) {
        if [self.row, self.col] != [input_field.row, input_field.col] || [self.row, self.col] != [streaming_weight.row, streaming_weight.col] {
            panic!("panicked at line {} in {}", line!(), file!());
        }
        if self.margin != streaming_weight.margin {
            panic!("panicked at line {} in {}", line!(), file!());
        }
        let margin = self.margin as i32;
        let row = self.row as i32;
        let col = self.col as i32;
        self.u_vert.slice_mut(s![margin..row-margin, margin..col-margin]).fill(0.0);
        self.u_hori.slice_mut(s![margin..row-margin, margin..col-margin]).fill(0.0);
        self.rho.slice_mut(s![margin..row-margin, margin..col-margin]).fill(0.0);

        for dr in -1..=1_i32 {
            for dc in -1..=1_i32 {
                let f_slice = self.f.slice_mut(s![margin..row-margin, margin..col-margin, dr+1, dc+1]);
                let w0_slice = streaming_weight.w0.slice(s![margin..row-margin, margin..col-margin, dr+1, dc+1]);
                let w1_slice = streaming_weight.w1.slice(s![margin..row-margin, margin..col-margin, dr+1, dc+1]);
                let f_prev_slice = input_field.f.slice(s![margin-dr..row-dr-margin, margin-dc..col-dc-margin, dr+1, dc+1]);
                Zip::from(f_slice).and(w0_slice).and(w1_slice).and(f_prev_slice)
                    .for_each(|f, w0, w1, f_prev| {
                        *f = w0 + w1 * f_prev;
                    });
                
                let f_slice = self.f.slice(s![margin..row-margin, margin..col-margin, dr+1, dc+1]);
                let mut u_vert_slice = self.u_vert.slice_mut(s![margin..row-margin, margin..col-margin]);
                let mut u_hori_slice = self.u_hori.slice_mut(s![margin..row-margin, margin..col-margin]);
                let mut rho_slice = self.rho.slice_mut(s![margin..row-margin, margin..col-margin]);
                Zip::from(f_slice).and(u_vert_slice).and(u_hori_slice).and(rho_slice)
                    .for_each(|f, u_vert, u_hori, rho|{
                        *u_vert += f * dr as f64;
                        *u_hori += f * dc as f64;
                        *rho += f;
                    });
            }
        }

        let mut u_vert_slice = self.u_vert.slice_mut(s![margin..row-margin, margin..col-margin]);
        let mut u_hori_slice = self.u_hori.slice_mut(s![margin..row-margin, margin..col-margin]);
        let rho_slice = self.rho.slice(s![margin..row-margin, margin..col-margin]);
        Zip::from(u_vert_slice).and(u_hori_slice).and(rho_slice).for_each(|u_vert, u_hori, rho| {
            *u_vert /= rho;
            *u_hori /= rho;
        });
    }
}

macro_rules! assert_delta {
    ($x:expr, $y:expr, $d:expr) => {
        if !($x - $y < $d && $y - $x < $d) { panic!("left: {}, right: {}", $x, $y); }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_input_field_set(){
        /*
        f
        ... ..2
        .1. ...
        ... ...
        
        ... ...
        ... 3..
        ... ...
         */
        let mut input_field = InputField::new(2, 2);
        let u_vert = arr2(&[[0.2, 0.4], [-0.3, -0.2]]);
        let u_hori = arr2(&[[-0.2, -0.1], [0.2, 0.2]]);
        let rho = arr2(&[[1.0, 0.8], [0.9, 1.1]]);
        input_field.set(u_vert, u_hori, rho);
        assert_delta!( 0.39111111111111111111111, *input_field.f.get((0, 0, 1, 1)).unwrap(), ERROR_DELTA ); // 1
        assert_delta!( 0.00822222222222222222222, *input_field.f.get((0, 1, 0, 2)).unwrap(), ERROR_DELTA ); // 2
        assert_delta!( 0.05622222222222222222222, *input_field.f.get((1, 1, 1, 0)).unwrap(), ERROR_DELTA ); // 3
    }

    #[test]
    fn test_streamed_field_stream(){
        let mut input_field = InputField::new(3, 3);
        input_field.f = Array::range(1., 81.5, 1.).into_shape((3, 3, 3, 3)).unwrap();
        let mut streaming_weight = StreamingWeight::new(3, 3, 1);
        let mut streamed_field = StreamedField::new(3, 3, 1);
        streaming_weight.w0 = streaming_weight.w0 + Array::range(0., 40.2, 0.5).into_shape((3, 3, 3, 3)).unwrap();
        streaming_weight.w1 = streaming_weight.w1 + Array::range(81., 0.5, -1.).into_shape((3, 3, 3, 3)).unwrap(); // あえて足していることに注意
        streamed_field.stream(&input_field, &streaming_weight);
        // println!("{}", streamed_field.f);
        // println!("{}", streamed_field.u_vert);
        // println!("{}", streamed_field.u_hori);
        // println!("{}", streamed_field.rho);
        for r in 0..=2 {
            for c in 0..=2 {
                if r == 1 && c == 1 { continue; }
                assert!( streamed_field.u_vert.get((r, c)).unwrap().is_nan() );
                assert!( streamed_field.u_hori.get((r, c)).unwrap().is_nan() );
                assert!( streamed_field.rho.get((r, c)).unwrap().is_nan() );

                for dr in 0..=2 {
                    for dc in 0..=2 {
                        assert!( streamed_field.f.get((r, c, dr, dc)).unwrap().is_nan() );
                    }
                }
            }
        }
        assert_delta!( *streamed_field.f.get((1, 1, 1, 1)).unwrap(), 1742.0, ERROR_DELTA );
        assert_delta!( *streamed_field.f.get((1, 1, 0, 2)).unwrap(), 2527.0, ERROR_DELTA );
        assert_delta!( *streamed_field.f.get((1, 1, 1, 0)).unwrap(), 2126.5, ERROR_DELTA );
        assert_delta!( *streamed_field.rho.get((1, 1)).unwrap(), 16158.0, ERROR_DELTA );
        assert_delta!( *streamed_field.u_vert.get((1, 1)).unwrap(), -0.41942072038, ERROR_DELTA );
        assert_delta!( *streamed_field.u_hori.get((1, 1)).unwrap(), -0.13980690679, ERROR_DELTA );
    }
}