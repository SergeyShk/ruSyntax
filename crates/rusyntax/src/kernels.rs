//! Ядра прямого прохода: умножение матрицы на векторы плитками и ячейка LSTM
//!
//! Взяты из ruMorph 0.6.0 (`crates/rumorph/src/dictionary/context.rs`, тот же автор, MIT); добавлены скалярное
//! произведение для оценки вершин и BiLSTM для нескольких предложений сразу.

/// Матрица f32 по строкам
#[derive(Debug)]
pub(crate) struct Matrix {
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) data: Vec<f32>,
}

impl Matrix {
    fn row(&self, row: usize) -> &[f32] {
        &self.data[row * self.cols..(row + 1) * self.cols]
    }
}

/// Число строк плитки `Tiled`
const TILE: usize = 16;
/// Число векторов, которые плитка умножает за раз: `Tiled::apply_all` держит столько сумм
pub(crate) const BATCH: usize = 4;
const _: () = assert!(BATCH == 4);

/// Матрица по плиткам из `TILE` строк для умножения на много векторов: в плитке столбцы подряд,
/// в столбце - числа `TILE` строк. Суммы плитки на `BATCH` векторов держатся в регистрах,
/// а каждое число матрицы читается один раз на `BATCH` векторов
#[derive(Debug)]
pub(crate) struct Tiled {
    rows: usize,
    cols: usize,
    columns: Vec<[f32; TILE]>,
}

impl Tiled {
    pub(crate) fn new(matrix: &Matrix) -> Self {
        let tiles = matrix.rows.div_ceil(TILE);
        let mut columns = vec![[0.0; TILE]; tiles * matrix.cols];
        for row in 0..matrix.rows {
            let (tile, offset) = (row / TILE, row % TILE);
            for (col, &value) in matrix.row(row).iter().enumerate() {
                columns[tile * matrix.cols + col][offset] = value;
            }
        }
        Tiled {
            rows: matrix.rows,
            cols: matrix.cols,
            columns,
        }
    }

    /// `out += self · input` для одного вектора
    pub(crate) fn apply_add(&self, input: &[f32], out: &mut [f32]) {
        #[cfg(target_arch = "x86_64")]
        if fused() {
            // SAFETY: AVX2 и FMA у процессора есть - проверено `fused`
            return unsafe { self.apply_add_fused(input, out) };
        }
        self.apply_add_with::<false>(input, out);
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    fn apply_add_fused(&self, input: &[f32], out: &mut [f32]) {
        self.apply_add_with::<true>(input, out);
    }

    #[inline(always)]
    fn apply_add_with<const FUSED: bool>(&self, input: &[f32], out: &mut [f32]) {
        // Две плитки за проход: две независимые цепочки сумм не ждут друг друга; порядок сложений
        // в каждой строке тот же, что по одной плитке
        let mut tiles = self.columns.chunks_exact(self.cols).enumerate();
        while let Some((tile, first)) = tiles.next() {
            let Some((_, second)) = tiles.next() else {
                let mut sums = [0.0f32; TILE];
                for (column, &value) in first.iter().zip(input) {
                    for j in 0..TILE {
                        sums[j] = madd::<FUSED>(value, column[j], sums[j]);
                    }
                }
                self.add_tile(tile, &sums, out);
                break;
            };
            let (mut a, mut b) = ([0.0f32; TILE], [0.0f32; TILE]);
            for ((ca, cb), &value) in first.iter().zip(second).zip(input) {
                for j in 0..TILE {
                    a[j] = madd::<FUSED>(value, ca[j], a[j]);
                    b[j] = madd::<FUSED>(value, cb[j], b[j]);
                }
            }
            self.add_tile(tile, &a, out);
            self.add_tile(tile + 1, &b, out);
        }
    }

    /// `out` строк плитки `tile` += суммы
    #[inline(always)]
    fn add_tile(&self, tile: usize, sums: &[f32; TILE], out: &mut [f32]) {
        let start = tile * TILE;
        let end = (start + TILE).min(self.rows);
        for (out, sum) in out[start..end].iter_mut().zip(sums) {
            *out += sum;
        }
    }

    /// `out[t] = self · inputs[t] + bias` для всех векторов `inputs` подряд
    pub(crate) fn apply_all(&self, inputs: &[f32], bias: &[f32], out: &mut [f32]) {
        #[cfg(target_arch = "x86_64")]
        if fused() {
            // SAFETY: AVX2 и FMA у процессора есть - проверено `fused`
            return unsafe { self.apply_all_fused(inputs, bias, out) };
        }
        self.apply_all_with::<false>(inputs, bias, out);
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    fn apply_all_fused(&self, inputs: &[f32], bias: &[f32], out: &mut [f32]) {
        self.apply_all_with::<true>(inputs, bias, out);
    }

    #[inline(always)]
    fn apply_all_with<const FUSED: bool>(&self, inputs: &[f32], bias: &[f32], out: &mut [f32]) {
        let count = inputs.len() / self.cols;
        let mut padded = vec![[0.0f32; BATCH]; self.cols];
        for first in (0..count).step_by(BATCH) {
            let batch = (count - first).min(BATCH);
            // Векторы пакета по столбцам: столбец - числа `BATCH` векторов
            for (col, values) in padded.iter_mut().enumerate() {
                for (b, value) in values.iter_mut().enumerate() {
                    *value = if b < batch {
                        inputs[(first + b) * self.cols + col]
                    } else {
                        0.0
                    };
                }
            }
            for (tile, columns) in self.columns.chunks_exact(self.cols).enumerate() {
                // Четыре отдельные суммы, а не массив: так они остаются в регистрах при любом
                // профиле сборки (с несколькими единицами кодогенерации массив уходил в память)
                let (mut s0, mut s1, mut s2, mut s3) =
                    ([0.0f32; TILE], [0.0; TILE], [0.0; TILE], [0.0; TILE]);
                for (column, values) in columns.iter().zip(&padded) {
                    axpy::<FUSED>(&mut s0, column, values[0]);
                    axpy::<FUSED>(&mut s1, column, values[1]);
                    axpy::<FUSED>(&mut s2, column, values[2]);
                    axpy::<FUSED>(&mut s3, column, values[3]);
                }
                let sums = [s0, s1, s2, s3];
                for (b, sums) in sums.iter().enumerate().take(batch) {
                    for (j, sum) in sums.iter().enumerate() {
                        let row = tile * TILE + j;
                        if row < self.rows {
                            out[(first + b) * self.rows + row] = sum + bias[row];
                        }
                    }
                }
            }
        }
    }
}

/// `sums += value · column`
#[inline(always)]
fn axpy<const FUSED: bool>(sums: &mut [f32; TILE], column: &[f32; TILE], value: f32) {
    for (sum, &weight) in sums.iter_mut().zip(column) {
        *sum = madd::<FUSED>(value, weight, *sum);
    }
}

/// `a · b + c` одной операцией FMA, где она есть: на aarch64 всегда, на x86_64 - в ядрах,
/// собранных с FMA (`FUSED`). Это на четверть быстрее, а последние биты суммы - другие, чем без FMA;
/// без нее `mul_add` - вызов библиотеки
#[inline(always)]
fn madd<const FUSED: bool>(a: f32, b: f32, c: f32) -> f32 {
    if FUSED || cfg!(target_arch = "aarch64") {
        a.mul_add(b, c)
    } else {
        a * b + c
    }
}

/// Есть ли у процессора AVX2 и FMA: ядра умножения матриц собраны еще и с ними, а колеса x86_64 -
/// без них, чтобы работать на любом процессоре. Ответ макрос держит в памяти
#[cfg(target_arch = "x86_64")]
fn fused() -> bool {
    std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
}

/// e^x с относительной ошибкой около 2e-7: без libm и ветвлений, поэтому цикл по массиву
/// векторизуется. Округление - сложением 1,5·2²³ (SSE2 не округляет векторно)
#[inline(always)]
pub(crate) fn exp(x: f32) -> f32 {
    const ROUND: f32 = 12_582_912.0;
    let x = x.clamp(-87.0, 88.0);
    let n = (x * std::f32::consts::LOG2_E + ROUND) - ROUND;
    // ln 2 двумя частями: остаток точен и при больших n
    let r = x - n * 0.693_145_75 - n * 1.428_606_8e-6;
    let p = 1.0
        + r * (1.0
            + r * (0.5 + r * (1.0 / 6.0 + r * (1.0 / 24.0 + r * (1.0 / 120.0 + r / 720.0)))));
    p * f32::from_bits(((n as i32 + 127) as u32) << 23)
}

#[inline(always)]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + exp(-x))
}

#[inline(always)]
fn tanh(x: f32) -> f32 {
    2.0 / (1.0 + exp(-2.0 * x)) - 1.0
}

/// Ячейка LSTM одного направления: матрицы ворот в порядке PyTorch (i, f, g, o), смещения сложены
#[derive(Debug)]
pub(crate) struct Lstm {
    pub(crate) input: Tiled,
    pub(crate) hidden: Tiled,
    pub(crate) bias: Vec<f32>,
}

/// Скалярное произведение блоками по 8: так цикл векторизуется
pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut sums = [0.0f32; 8];
    let (left, right) = (a.as_chunks::<8>(), b.as_chunks::<8>());
    for (x, y) in left.0.iter().zip(right.0) {
        for i in 0..8 {
            sums[i] += x[i] * y[i];
        }
    }
    let tail: f32 = left.1.iter().zip(right.1).map(|(x, y)| x * y).sum();
    sums.iter().sum::<f32>() + tail
}

/// Шаг ячейки: `gates` - суммы ворот i, f, g, o (вход, состояние и смещение), обновляются `c` и `h`
#[inline(always)]
fn step(gates: &mut [f32], h: &mut [f32], c: &mut [f32]) {
    let hidden = h.len();
    // Каждый цикл - по массиву, без ветвлений
    let (sigmoids, rest) = gates.split_at_mut(2 * hidden);
    let (candidates, outputs) = rest.split_at_mut(hidden);
    sigmoids
        .iter_mut()
        .for_each(|value| *value = sigmoid(*value));
    candidates
        .iter_mut()
        .for_each(|value| *value = tanh(*value));
    outputs
        .iter_mut()
        .for_each(|value| *value = sigmoid(*value));
    let (inputs, forgets) = sigmoids.split_at(hidden);
    for i in 0..hidden {
        c[i] = forgets[i] * c[i] + inputs[i] * candidates[i];
    }
    for i in 0..hidden {
        h[i] = outputs[i] * tanh(c[i]);
    }
}

/// Двунаправленный слой для одного предложения из `n` векторов: выход - `n × d`, половины - прямое
/// и обратное направления
pub(crate) fn bilstm(layer: &[Lstm; 2], input: &[f32], n: usize, d: usize) -> Vec<f32> {
    let hidden = d / 2;
    let mut output = vec![0.0f32; n * d];
    // Вклад входа во все ворота считается сразу для всех токенов
    let gates_input = layer.each_ref().map(|cell| {
        let mut gates = vec![0.0f32; n * 4 * hidden];
        cell.input.apply_all(input, &cell.bias, &mut gates);
        gates
    });
    let mut states = [(); 2].map(|()| (vec![0.0f32; hidden], vec![0.0f32; hidden]));
    let mut gates = [(); 2].map(|()| vec![0.0f32; 4 * hidden]);
    for t_step in 0..n {
        for direction in 0..2 {
            let t = if direction == 0 {
                t_step
            } else {
                n - 1 - t_step
            };
            let (h, c) = &mut states[direction];
            let gates = &mut gates[direction];
            gates.copy_from_slice(&gates_input[direction][t * 4 * hidden..(t + 1) * 4 * hidden]);
            layer[direction].hidden.apply_add(h, gates);
            step(gates, h, c);
            output[t * d + direction * hidden..t * d + (direction + 1) * hidden].copy_from_slice(h);
        }
    }
    output
}

/// Тот же слой для нескольких предложений (не больше `BATCH`): на каждом шаге рекуррентная матрица
/// умножается на состояния всех предложений сразу (`Tiled::apply_all`) и читается из памяти один раз.
/// Суммы ворот те же, что у `bilstm`, бит в бит: порядок сложений в плитке один
pub(crate) fn bilstm_many(
    layer: &[Lstm; 2],
    inputs: &[&[f32]],
    lens: &[usize],
    d: usize,
) -> Vec<Vec<f32>> {
    debug_assert!(inputs.len() <= BATCH && inputs.len() == lens.len());
    let hidden = d / 2;
    let count = inputs.len();
    let longest = lens.iter().copied().max().unwrap_or(0);
    let mut outputs: Vec<Vec<f32>> = lens.iter().map(|&n| vec![0.0f32; n * d]).collect();
    let zero = vec![0.0f32; 4 * hidden];
    for (direction, cell) in layer.iter().enumerate() {
        let gates_input: Vec<Vec<f32>> = inputs
            .iter()
            .zip(lens)
            .map(|(input, &n)| {
                let mut gates = vec![0.0f32; n * 4 * hidden];
                cell.input.apply_all(input, &cell.bias, &mut gates);
                gates
            })
            .collect();
        let mut h = vec![0.0f32; count * hidden];
        let mut c = vec![0.0f32; count * hidden];
        let mut sums = vec![0.0f32; count * 4 * hidden];
        for t_step in 0..longest {
            cell.hidden.apply_all(&h, &zero, &mut sums);
            for (s, &n) in lens.iter().enumerate() {
                if t_step >= n {
                    continue;
                }
                let t = if direction == 0 {
                    t_step
                } else {
                    n - 1 - t_step
                };
                let gates = &mut sums[s * 4 * hidden..(s + 1) * 4 * hidden];
                for (gate, input) in gates
                    .iter_mut()
                    .zip(&gates_input[s][t * 4 * hidden..(t + 1) * 4 * hidden])
                {
                    *gate += input;
                }
                let (hs, cs) = (
                    &mut h[s * hidden..(s + 1) * hidden],
                    &mut c[s * hidden..(s + 1) * hidden],
                );
                step(gates, hs, cs);
                outputs[s][t * d + direction * hidden..t * d + (direction + 1) * hidden]
                    .copy_from_slice(hs);
            }
        }
    }
    outputs
}
