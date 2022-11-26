use ndarray::{Array2, Array4, arr2, stack, Axis, Zip, s};
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
const ERROR_DELTA: f64 = 0.00000000000001;

pub struct InputField {
    row: usize,
    col: usize,
    f: Array4<f64>,
    u_vert: Array2<f64>, // 計算の都合上4Darrayとする(最後の(3, 3)は同じ値)
    u_hori: Array2<f64>,
    // u2: Array2<f64>,
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
}