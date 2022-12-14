use ndarray::{Array, Array2, Array4, arr0, arr2, stack, Axis, Zip, s, azip};
use ndarray_parallel::prelude::*;
use std::f64::NAN;

// 計算できない値についてはNaNを入れる
// 外積(v x u)はdr * u_hori - dc * u_vert

// 命名規則(必ず新しい規則はここに書く)
// 1->2->3の順に書いていく eg. u_hori_nxnx
// 1. f:粒子密度  feq:eq場の粒子密度  u:風速  rho:密度
// 1. w0,w1,w2,w3,w4:そのレイヤーの重み  dw0,dw1,dw2,dw3,dw4:重みの変化分
// 2. _vert:縦,緯線方向(下向き正！)  _hori:横,経線方向(右向き正)
// 3. _next:次の  _now:今の(例えばCollidingWeight->CollidedFieldという流れならWeightに対するfield)  _prev:前の
// Weight(prev) -> Field(prev) -> Weight(now, あるいは添字なし) -> Field(now あるいは添字なし) -> Weight(next) -> Field(next)
// row:行数  col:列数  r:r行(添字)  c:c列(添字) dr, dc
// C:係数

// TODO: 速度改善のために、[dr, dc, r, c]の順にするべきかも 遅かったら後で試してみる
// TODO: fからu_vert, u_hori, rhoを計算するところは共通化できそう
// TODO: それぞれの構造体がいまどういう状態か(stream()したか、collide()したか、重みの更新を行ったか)を記録して、不正な状態遷移を防ぐ

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
    dw0: Array4<f64>,
    dw1: Array4<f64>,
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

pub struct CollidingWeight {
    row: usize,
    col: usize,
    margin: usize,
    w1: Array4<f64>,
    w2: Array4<f64>,
    w3: Array4<f64>,
    w4: Array4<f64>,
    dw1: Array4<f64>,
    dw2: Array4<f64>,
    dw3: Array4<f64>,
    dw4: Array4<f64>,
    delta: Array4<f64>,
}

pub struct CollidedField {
    row: usize,
    col: usize,
    margin: usize,
    f: Array4<f64>,
    // u_vert: Array2<f64>, // 多分必要ないので今のところコメントアウトしておく
    // u_hori: Array2<f64>,
    // rho: Array2<f64>,
    feq: Array4<f64>,
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
                let mut f_slice = self.f.slice_mut(s![.., .., dr+1, dc+1]);
                Zip::from(&mut f_slice).and(&self.u_vert).and(&self.u_hori).and(&self.rho)
                    .for_each(|f, u_vert, u_hori, rho| {
                        let dr_f = dr as f64; // -1., 0., 1.のいずれかに変換
                        let dc_f = dc as f64;
                        let u2 = u_vert * u_vert + u_hori * u_hori; // ここちょっと無駄な気もする
                        let u_prod = u_vert * dr_f + u_hori * dc_f;
                        *f = C[(dr+1) as usize][(dc+1) as usize] * rho * (1.0 + (3.0 + 4.5 * u_prod) * u_prod - 1.5 * u2);
                    });
            }
        }
    }
}

impl StreamingWeight {
    pub fn new(row: usize, col: usize, margin: usize) -> StreamingWeight {
        let mut w0 = Array4::<f64>::from_elem((row, col, 3, 3), NAN);
        let mut w1 = Array4::<f64>::from_elem((row, col, 3, 3), NAN);
        let mut dw0 = Array4::<f64>::from_elem((row, col, 3, 3), NAN);
        let mut dw1 = Array4::<f64>::from_elem((row, col, 3, 3), NAN);
        let mut delta = Array4::<f64>::from_elem((row, col, 3, 3), NAN);
        w0.slice_mut(s![margin..row-margin, margin..col-margin, .., ..]).fill(0.0);
        w1.slice_mut(s![margin..row-margin, margin..col-margin, .., ..]).fill(1.0);
        dw0.slice_mut(s![margin..row-margin, margin..col-margin, .., ..]).fill(0.0);
        dw1.slice_mut(s![margin..row-margin, margin..col-margin, .., ..]).fill(0.0);
        delta.slice_mut(s![margin..row-margin, margin..col-margin, .., ..]).fill(0.0);
        StreamingWeight { row, col, margin, w0, w1, dw0, dw1, delta }
    }

    pub fn propagate_from_output(self: &mut Self, eta: f64, field_now: &StreamedField, field_prev: &CollidedField, u_vert_ans: &Array2<f64>, u_hori_ans: &Array2<f64>) {
        let shape = [self.col, self.row];
        if shape != [field_now.col, field_now.row] || shape != [field_prev.col, field_prev.col] || shape != u_vert_ans.shape() || shape != u_hori_ans.shape() {
            panic!("panicked at line {} in {}", line!(), file!());
        }
        if self.margin != field_now.margin || self.margin != field_prev.margin + 1 {
            panic!("panicked at line {} in {}", line!(), file!());
        }

        // rho_now_inv
        let row = self.row as i32;
        let col = self.col as i32;
        let margin = self.margin as i32;
        let mut inv_rho_now = Array2::<f64>::from_elem((self.row, self.col), NAN);
        Zip::from(&mut inv_rho_now.slice_mut(s![margin..row-margin, margin..col-margin]))
            .and(&field_now.rho.slice(s![margin..row-margin, margin..col-margin]))
            .for_each(|inv_rho_now, rho_now| {
                *inv_rho_now = 1.0 / rho_now;
            });
        for dr in -1..=1_i32 {
            for dc in -1..=1_i32 {
                Zip::from(&mut self.delta.slice_mut(s![margin..row-margin, margin..col-margin, dr+1, dc+1]))
                    .and(&inv_rho_now.slice(s![margin..row-margin, margin..col-margin]))
                    .and(&field_now.u_vert.slice(s![margin..row-margin, margin..col-margin]))
                    .and(&field_now.u_hori.slice(s![margin..row-margin, margin..col-margin]))
                    .and(&u_vert_ans.slice(s![margin..row-margin, margin..col-margin]))
                    .and(&u_hori_ans.slice(s![margin..row-margin, margin..col-margin]))
                    .for_each(|delta, inv_rho_now, u_vert_now, u_hori_now, u_vert_ans, u_hori_ans|{
                        *delta = inv_rho_now * ((u_vert_now - u_vert_ans) * (dr as f64 - u_vert_now) + (u_hori_now - u_hori_ans) * (dc as f64 - u_hori_now));
                    });

                Zip::from(&mut self.dw0.slice_mut(s![margin..row-margin, margin..col-margin, dr+1, dc+1]))
                    .and(&mut self.dw1.slice_mut(s![margin..row-margin, margin..col-margin, dr+1, dc+1]))
                    .and(&self.delta.slice(s![margin..row-margin, margin..col-margin, dr+1, dc+1]))
                    .and(&field_prev.f.slice(s![margin-dr..row-dr-margin, margin-dc..col-dc-margin, dr+1, dc+1]))
                    .for_each(|dw0, dw1, delta, f_prev|{
                        *dw0 = -eta * delta;
                        *dw1 = *dw0 * f_prev;
                    });
            }
        }
    }

    pub fn update(self: &mut Self) {
        let margin = self.margin;
        let row = self.row;
        let col = self.col;
        let mut w0_slice = self.w0.slice_mut(s![margin..row-margin, margin..col-margin, .., ..]);
        let mut w1_slice = self.w1.slice_mut(s![margin..row-margin, margin..col-margin, .., ..]);
        let dw0_slice = self.dw0.slice(s![margin..row-margin, margin..col-margin, .., ..]);
        let dw1_slice = self.dw1.slice(s![margin..row-margin, margin..col-margin, .., ..]);
        Zip::from(&mut w0_slice).and(&dw0_slice).for_each(|w0, dw0|{ *w0 = *w0 + dw0; });
        Zip::from(&mut w1_slice).and(&dw1_slice).for_each(|w1, dw1|{ *w1 = *w1 + dw1; });
        self.dw0.slice_mut(s![margin..row-margin, margin..col-margin, .., ..]).fill(0.0);
        self.dw1.slice_mut(s![margin..row-margin, margin..col-margin, .., ..]).fill(0.0);
    }
}

impl StreamedField {
    pub fn new(row: usize, col: usize, margin: usize) -> StreamedField {
        let mut f = Array4::<f64>::from_elem((row, col, 3, 3), NAN);
        let mut u_vert = Array2::<f64>::from_elem((row, col), NAN);
        let mut u_hori = Array2::<f64>::from_elem((row, col), NAN);
        let mut rho = Array2::<f64>::from_elem((row, col), NAN);
        f.slice_mut(s![margin..row-margin, margin..col-margin, .., ..]).fill(0.0);
        u_vert.slice_mut(s![margin..row-margin, margin..col-margin]).fill(0.0);
        u_hori.slice_mut(s![margin..row-margin, margin..col-margin]).fill(0.0);
        rho.slice_mut(s![margin..row-margin, margin..col-margin]).fill(0.0);
        StreamedField { row, col, margin, f, u_vert, u_hori, rho }
    }

    pub fn stream_from_input_field(self: &mut Self, input_field: &InputField, streaming_weight: &StreamingWeight) {
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
                let mut f_slice = self.f.slice_mut(s![margin..row-margin, margin..col-margin, dr+1, dc+1]);
                let w0_slice = streaming_weight.w0.slice(s![margin..row-margin, margin..col-margin, dr+1, dc+1]);
                let w1_slice = streaming_weight.w1.slice(s![margin..row-margin, margin..col-margin, dr+1, dc+1]);
                let f_prev_slice = input_field.f.slice(s![margin-dr..row-dr-margin, margin-dc..col-dc-margin, dr+1, dc+1]);
                Zip::from(&mut f_slice).and(&w0_slice).and(&w1_slice).and(&f_prev_slice)
                    .for_each(|f, w0, w1, f_prev| {
                        *f = w0 + w1 * f_prev;
                    });
                
                let f_slice = self.f.slice(s![margin..row-margin, margin..col-margin, dr+1, dc+1]);
                let mut u_vert_slice = self.u_vert.slice_mut(s![margin..row-margin, margin..col-margin]);
                let mut u_hori_slice = self.u_hori.slice_mut(s![margin..row-margin, margin..col-margin]);
                let mut rho_slice = self.rho.slice_mut(s![margin..row-margin, margin..col-margin]);
                Zip::from(&f_slice).and(&mut u_vert_slice).and(&mut u_hori_slice).and(&mut rho_slice)
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
        Zip::from(&mut u_vert_slice).and(&mut u_hori_slice).and(&rho_slice).for_each(|u_vert, u_hori, rho| {
            *u_vert /= rho;
            *u_hori /= rho;
        });
    }

    pub fn stream_from_collided_field(self: &mut Self, collided_field: &CollidedField, streaming_weight: &StreamingWeight) {
        if [self.row, self.col] != [collided_field.row, collided_field.col] || [self.row, self.col] != [streaming_weight.row, streaming_weight.col] {
            panic!("panicked at line {} in {}", line!(), file!());
        }
        if self.margin != streaming_weight.margin || self.margin != collided_field.margin + 1 {
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
                let mut f_slice = self.f.slice_mut(s![margin..row-margin, margin..col-margin, dr+1, dc+1]);
                let w0_slice = streaming_weight.w0.slice(s![margin..row-margin, margin..col-margin, dr+1, dc+1]);
                let w1_slice = streaming_weight.w1.slice(s![margin..row-margin, margin..col-margin, dr+1, dc+1]);
                let f_prev_slice = collided_field.f.slice(s![margin-dr..row-dr-margin, margin-dc..col-dc-margin, dr+1, dc+1]);
                Zip::from(&mut f_slice).and(&w0_slice).and(&w1_slice).and(&f_prev_slice)
                    .for_each(|f, w0, w1, f_prev| {
                        *f = w0 + w1 * f_prev;
                    });
                
                let f_slice = self.f.slice(s![margin..row-margin, margin..col-margin, dr+1, dc+1]);
                let mut u_vert_slice = self.u_vert.slice_mut(s![margin..row-margin, margin..col-margin]);
                let mut u_hori_slice = self.u_hori.slice_mut(s![margin..row-margin, margin..col-margin]);
                let mut rho_slice = self.rho.slice_mut(s![margin..row-margin, margin..col-margin]);
                Zip::from(&f_slice).and(&mut u_vert_slice).and(&mut u_hori_slice).and(&mut rho_slice)
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
        Zip::from(&mut u_vert_slice).and(&mut u_hori_slice).and(&rho_slice).for_each(|u_vert, u_hori, rho| {
            *u_vert /= rho;
            *u_hori /= rho;
        });
    }
}

impl CollidingWeight {
    pub fn new(row: usize, col: usize, margin: usize) -> CollidingWeight {
        let mut w1 = Array4::<f64>::from_elem((row, col, 3, 3), NAN);
        let mut w2 = Array4::<f64>::from_elem((row, col, 3, 3), NAN);
        let mut w3 = Array4::<f64>::from_elem((row, col, 3, 3), NAN);
        let mut w4 = Array4::<f64>::from_elem((row, col, 3, 3), NAN);
        let mut dw1 = Array4::<f64>::from_elem((row, col, 3, 3), NAN);
        let mut dw2 = Array4::<f64>::from_elem((row, col, 3, 3), NAN);
        let mut dw3 = Array4::<f64>::from_elem((row, col, 3, 3), NAN);
        let mut dw4 = Array4::<f64>::from_elem((row, col, 3, 3), NAN);
        let mut delta = Array4::<f64>::from_elem((row, col, 3, 3), NAN);
        w1.slice_mut(s![margin..row-margin, margin..col-margin, .., ..]).fill(3.0);
        w2.slice_mut(s![margin..row-margin, margin..col-margin, .., ..]).fill(0.0);
        w3.slice_mut(s![margin..row-margin, margin..col-margin, .., ..]).fill(4.5);
        w4.slice_mut(s![margin..row-margin, margin..col-margin, .., ..]).fill(-1.5);
        dw1.slice_mut(s![margin..row-margin, margin..col-margin, .., ..]).fill(0.0);
        dw2.slice_mut(s![margin..row-margin, margin..col-margin, .., ..]).fill(0.0);
        dw3.slice_mut(s![margin..row-margin, margin..col-margin, .., ..]).fill(0.0);
        dw4.slice_mut(s![margin..row-margin, margin..col-margin, .., ..]).fill(0.0);
        delta.slice_mut(s![margin..row-margin, margin..col-margin, .., ..]).fill(0.0);
        CollidingWeight { row, col, margin, w1, w2, w3, w4, dw1, dw2, dw3, dw4, delta }
    }

    pub fn update(self: &mut Self) {
        let margin = self.margin;
        let row = self.row;
        let col = self.col;
        let mut w1_slice = self.w1.slice_mut(s![margin..row-margin, margin..col-margin, .., ..]);
        let mut w2_slice = self.w2.slice_mut(s![margin..row-margin, margin..col-margin, .., ..]);
        let mut w3_slice = self.w3.slice_mut(s![margin..row-margin, margin..col-margin, .., ..]);
        let mut w4_slice = self.w4.slice_mut(s![margin..row-margin, margin..col-margin, .., ..]);
        let dw1_slice = self.dw1.slice(s![margin..row-margin, margin..col-margin, .., ..]);
        let dw2_slice = self.dw2.slice(s![margin..row-margin, margin..col-margin, .., ..]);
        let dw3_slice = self.dw3.slice(s![margin..row-margin, margin..col-margin, .., ..]);
        let dw4_slice = self.dw4.slice(s![margin..row-margin, margin..col-margin, .., ..]);
        Zip::from(&mut w1_slice).and(&dw1_slice).for_each(|w1, dw1|{ *w1 = *w1 + dw1; });
        Zip::from(&mut w2_slice).and(&dw2_slice).for_each(|w2, dw2|{ *w2 = *w2 + dw2; });
        Zip::from(&mut w3_slice).and(&dw3_slice).for_each(|w3, dw3|{ *w3 = *w3 + dw3; });
        Zip::from(&mut w4_slice).and(&dw4_slice).for_each(|w4, dw4|{ *w4 = *w4 + dw4; });
        self.dw1.slice_mut(s![margin..row-margin, margin..col-margin, .., ..]).fill(0.0);
        self.dw2.slice_mut(s![margin..row-margin, margin..col-margin, .., ..]).fill(0.0);
        self.dw3.slice_mut(s![margin..row-margin, margin..col-margin, .., ..]).fill(0.0);
        self.dw4.slice_mut(s![margin..row-margin, margin..col-margin, .., ..]).fill(0.0);
    }
}

impl CollidedField {
    pub fn new(row: usize, col: usize, margin: usize) -> CollidedField {
        let mut f = Array4::<f64>::from_elem((row, col, 3, 3), NAN);
        let mut feq = Array4::<f64>::from_elem((row, col, 3, 3), NAN);
        f.slice_mut(s![margin..row-margin, margin..col-margin, .., ..]).fill(0.0);
        feq.slice_mut(s![margin..row-margin, margin..col-margin, .., ..]).fill(0.0);
        CollidedField { row, col, margin, f, feq }
    }

    pub fn collide(self: &mut Self, streamed_field: &StreamedField, colliding_weight: &CollidingWeight) {
        if [self.row, self.col] != [streamed_field.row, streamed_field.col] || [self.row, self.col] != [colliding_weight.row, colliding_weight.col] {
            panic!("panicked at line {} in {}", line!(), file!());
        }
        if self.margin != streamed_field.margin || self.margin != colliding_weight.margin {
            panic!("panicked at line {} in {}", line!(), file!());
        }
        let margin = self.margin as i32;
        let row = self.row as i32;
        let col = self.col as i32;
        self.feq.slice_mut(s![margin..row-margin, margin..col-margin, .., ..]).fill(1.0); // zip from and and ...が6つまでしか繋いでくれないのでちょっと工夫

        for dr in -1..=1_i32 { 
            for dc in -1..=1_i32 {
                let mut feq_slice = self.feq.slice_mut(s![margin..row-margin, margin..col-margin, dr+1, dc+1]);
                let u_vert_prev_slice = streamed_field.u_vert.slice(s![margin..row-margin, margin..col-margin]);
                let u_hori_prev_slice = streamed_field.u_hori.slice(s![margin..row-margin, margin..col-margin]);
                let rho_prev_slice = streamed_field.rho.slice(s![margin..row-margin, margin..col-margin]);
                let w1_slice = colliding_weight.w1.slice(s![margin..row-margin, margin..col-margin, dr+1, dc+1]);
                let w2_slice = colliding_weight.w2.slice(s![margin..row-margin, margin..col-margin, dr+1, dc+1]);
                let w3_slice = colliding_weight.w3.slice(s![margin..row-margin, margin..col-margin, dr+1, dc+1]);
                let w4_slice = colliding_weight.w4.slice(s![margin..row-margin, margin..col-margin, dr+1, dc+1]);
                Zip::from(&mut feq_slice).and(&u_vert_prev_slice).and(&u_hori_prev_slice).and(&w1_slice).and(&w3_slice).for_each(|feq, u_vert_prev, u_hori_prev, w1, w3|{
                    let u_prod = u_vert_prev * dr as f64 + u_hori_prev * dc as f64;
                    *feq += (w3 * u_prod + w1) * u_prod;
                });
                Zip::from(&mut feq_slice).and(&u_vert_prev_slice).and(&u_hori_prev_slice).and(&w2_slice).for_each(|feq, u_vert_prev, u_hori_prev, w2|{
                    *feq += w2 * (dr as f64 * u_hori_prev - dc as f64 * u_vert_prev);
                });
                Zip::from(&mut feq_slice).and(&u_vert_prev_slice).and(&u_hori_prev_slice).and(&w4_slice).for_each(|feq, u_vert_prev, u_hori_prev, w4|{
                    let u2 = u_vert_prev * u_vert_prev + u_hori_prev * u_hori_prev;
                    *feq += w4 * u2;
                });
                Zip::from(&mut feq_slice).and(&rho_prev_slice).for_each(|feq, rho_prev|{
                    *feq *= C[(dr+1) as usize][(dc+1) as usize] * rho_prev;
                });
            }
        }
        
        let mut f_slice = self.f.slice_mut(s![margin..row-margin, margin..col-margin, .., ..]);
        let feq_slice = self.feq.slice(s![margin..row-margin, margin..col-margin, .., ..]);
        let f_prev_slice = streamed_field.f.slice(s![margin..row-margin, margin..col-margin, .., ..]);
        Zip::from(&mut f_slice).and(&feq_slice).and(&f_prev_slice).for_each(|f, feq, f_prev| {
            *f = (feq + f_prev) / 2.0;
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
        for _ in 0..5{
            let u_vert = arr2(&[[0.2, 0.4], [-0.3, -0.2]]);
            let u_hori = arr2(&[[-0.2, -0.1], [0.2, 0.2]]);
            let rho = arr2(&[[1.0, 0.8], [0.9, 1.1]]);
            input_field.set(u_vert, u_hori, rho);
            assert_delta!( 0.39111111111111111111111, *input_field.f.get((0, 0, 1, 1)).unwrap(), ERROR_DELTA ); // 1
            assert_delta!( 0.00822222222222222222222, *input_field.f.get((0, 1, 0, 2)).unwrap(), ERROR_DELTA ); // 2
            assert_delta!( 0.05622222222222222222222, *input_field.f.get((1, 1, 1, 0)).unwrap(), ERROR_DELTA ); // 3
        }
    }

    // テストは各メソッドについて、何度か流す(透過性のチェック)
    #[test]
    fn test_streamed_field_stream_from_input_field(){
        let mut input_field = InputField::new(3, 3);
        input_field.f = Array::range(1., 81.5, 1.).into_shape((3, 3, 3, 3)).unwrap();
        let mut streaming_weight = StreamingWeight::new(3, 3, 1);
        let mut streamed_field = StreamedField::new(3, 3, 1);
        streaming_weight.w0 = streaming_weight.w0 + Array::range(0., 40.2, 0.5).into_shape((3, 3, 3, 3)).unwrap();
        streaming_weight.w1 = streaming_weight.w1 + Array::range(81., 0.5, -1.).into_shape((3, 3, 3, 3)).unwrap(); // あえて足していることに注意
        for _ in 0..5 {
            streamed_field.stream_from_input_field(&input_field, &streaming_weight);
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

    #[test]
    fn test_stream_field_stream_from_collided_field() {
        let mut collided_field = CollidedField::new(3, 3, 0);
        collided_field.f = Array::range(1., 81.5, 1.).into_shape((3, 3, 3, 3)).unwrap();
        let mut streaming_weight = StreamingWeight::new(3, 3, 1);
        let mut streamed_field = StreamedField::new(3, 3, 1);
        streaming_weight.w0 = streaming_weight.w0 + Array::range(0., 40.2, 0.5).into_shape((3, 3, 3, 3)).unwrap();
        streaming_weight.w1 = streaming_weight.w1 + Array::range(81., 0.5, -1.).into_shape((3, 3, 3, 3)).unwrap(); // あえて足していることに注意
        for _ in 0..5 {
            streamed_field.stream_from_collided_field(&collided_field, &streaming_weight);
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

    #[test]
    fn test_collided_field_collide() {
        let mut collided_field = CollidedField::new(3, 3, 1);
        let mut colliding_weight = CollidingWeight::new(3, 3, 1);
        let mut streamed_field = StreamedField::new(3, 3, 1);
        streamed_field.f.slice_mut(s![1, 1, .., ..]).assign(&arr2(&[[1., 2., 3.], [6., 5., 4.], [7., 8., 9.]]));
        streamed_field.u_vert.slice_mut(s![1, 1]).assign(&arr0(0.4));
        streamed_field.u_hori.slice_mut(s![1, 1]).assign(&arr0(0.0444444444444444444));
        streamed_field.rho.slice_mut(s![1, 1]).assign(&arr0(45.));
        colliding_weight.w1 = colliding_weight.w1 + arr2(&[[0., 0.1, 0.2], [0.3, 0.4, 0.5], [0.6, 0.7, 0.8]]); // Nan + ... = Nan を利用
        colliding_weight.w2 = colliding_weight.w2 + arr2(&[[0.8, 0.7, 0.6], [0.5, 0.4, 0.3], [0.2, 0.1, 0.]]);
        colliding_weight.w3 = colliding_weight.w3 + arr2(&[[0.1, 0.4, 0.7], [0.2, 0.5, 0.8], [0.3, 0.6, 0.9]]);
        colliding_weight.w4 = colliding_weight.w4 + arr2(&[[0.9, 0.6, 0.3], [0.8, 0.5, 0.2], [0.7, 0.4, 0.1]]);

        for _ in 0..5 {
            collided_field.collide(&streamed_field, &colliding_weight);
            
            for r in 0..=2 {
                for c in 0..=2 {
                    if r == 1 && c == 1 { continue; }
                    // assert!( collided_field.u_vert.get((r, c)).unwrap().is_nan() );
                    // assert!( collided_field.u_hori.get((r, c)).unwrap().is_nan() );
                    // assert!( collided_field.rho.get((r, c)).unwrap().is_nan() );
    
                    for dr in 0..=2 {
                        for dc in 0..=2 {
                            assert!( collided_field.feq.get((r, c, dr, dc)).unwrap().is_nan() );
                        }
                    }
                }
            }
            assert_delta!( *collided_field.feq.get((1, 1, 0, 1)).unwrap(), 1.835555555555555, ERROR_DELTA );
            assert_delta!( *collided_field.feq.get((1, 1, 1, 1)).unwrap(), 16.760493827160495, ERROR_DELTA );
            assert_delta!( *collided_field.feq.get((1, 1, 1, 2)).unwrap(), 4.177283950617283, ERROR_DELTA );
            assert_delta!( *collided_field.feq.get((1, 1, 2, 0)).unwrap(), 3.5576543209876546, ERROR_DELTA );
            assert_delta!( *collided_field.f.get((1, 1, 0, 1)).unwrap(), 3.835555555555555 / 2.0, ERROR_DELTA );
            assert_delta!( *collided_field.f.get((1, 1, 1, 1)).unwrap(), 21.760493827160495 / 2.0, ERROR_DELTA );
            assert_delta!( *collided_field.f.get((1, 1, 1, 2)).unwrap(), 8.177283950617283 / 2.0, ERROR_DELTA );
            assert_delta!( *collided_field.f.get((1, 1, 2, 0)).unwrap(), 10.5576543209876546 / 2.0, ERROR_DELTA );
        }
    }

    #[test]
    fn test_streaming_weight_propagate_from_output() {
        let mut field_prev = CollidedField::new(3, 3, 0);
        let mut streaming_weight = StreamingWeight::new(3, 3, 1);
        let mut field_now = StreamedField::new(3, 3, 1);
        let eta = 0.1;
        let u_vert_ans = arr2(&[[NAN, NAN, NAN], [NAN, 0.2, NAN], [NAN, NAN, NAN]]);
        let u_hori_ans = arr2(&[[NAN, NAN, NAN], [NAN, 0.5, NAN], [NAN, NAN, NAN]]);
        field_now.u_vert.slice_mut(s![1, 1]).assign(&arr0( -2.0 / 45.0));
        field_now.u_hori.slice_mut(s![1, 1]).assign(&arr0( 6.0 / 45.0));
        field_now.rho.slice_mut(s![1, 1]).assign(&arr0(45.0));
        field_prev.f = Array::range(1., 81.5, 1.).into_shape((3, 3, 3, 3)).unwrap();

        for _ in 0..5 {
            streaming_weight.propagate_from_output(eta, &field_now, &field_prev, &u_vert_ans, &u_hori_ans);
            
            for r in 0..=2 {
                for c in 0..=2 {
                    if r == 1 && c == 1 { continue; }
    
                    for dr in 0..=2 {
                        for dc in 0..=2 {
                            assert!( streaming_weight.delta.get((r, c, dr, dc)).unwrap().is_nan() );
                            assert!( streaming_weight.w0.get((r, c, dr, dc)).unwrap().is_nan() );
                            assert!( streaming_weight.w1.get((r, c, dr, dc)).unwrap().is_nan() );
                        }
                    }
                }
            }

            assert_delta!( streaming_weight.delta.get((1, 1, 0, 1)).unwrap(), 0.00627709190672153635116, ERROR_DELTA );
            assert_delta!( streaming_weight.delta.get((1, 1, 1, 2)).unwrap(), -0.0073031550068587105624, ERROR_DELTA );
            assert_delta!( streaming_weight.delta.get((1, 1, 1, 1)).unwrap(), 0.00084499314128943758573, ERROR_DELTA );
            assert_delta!( streaming_weight.delta.get((1, 1, 2, 0)).unwrap(), 0.00356104252400548696844, ERROR_DELTA );

            assert_delta!( streaming_weight.dw0.get((1, 1, 0, 1)).unwrap(), -0.000627709190672153635116, ERROR_DELTA );
            assert_delta!( streaming_weight.dw0.get((1, 1, 1, 2)).unwrap(), 0.0007303155006858710562414, ERROR_DELTA );
            assert_delta!( streaming_weight.dw0.get((1, 1, 1, 1)).unwrap(), -0.000084499314128943758573, ERROR_DELTA );
            assert_delta!( streaming_weight.dw0.get((1, 1, 2, 0)).unwrap(), -0.000356104252400548696844, ERROR_DELTA );

            assert_delta!( streaming_weight.dw1.get((1, 1, 0, 1)).unwrap(), -0.0408010973936899862825788, ERROR_DELTA );
            assert_delta!( streaming_weight.dw1.get((1, 1, 1, 2)).unwrap(), 0.0241004115226337448559670, ERROR_DELTA );
            assert_delta!( streaming_weight.dw1.get((1, 1, 1, 1)).unwrap(), -0.0034644718792866941015089, ERROR_DELTA );
            assert_delta!( streaming_weight.dw1.get((1, 1, 2, 0)).unwrap(), -0.0089026063100137174211248, ERROR_DELTA );
        }
    }
}