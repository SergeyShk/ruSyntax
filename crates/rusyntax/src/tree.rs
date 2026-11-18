//! Дерево зависимостей по оценкам вершин: у каждого слова одна вершина, корень один, циклов нет.
//!
//! Сначала у каждого слова берется лучшая вершина; если это уже дерево с одним корнем, оно и есть дерево
//! максимального веса. Иначе - Чу - Лю - Эдмондс по логарифмам вероятностей вершин с перебором корня;
//! у предложений длиннее `FULL` слов (оценки всех пар заняли бы слишком много памяти) лишние корни и узлы
//! циклов подвешиваются к лучшему корню.

/// Длина предложения, до которой строится полная матрица оценок и ищется дерево максимального веса
pub(crate) const FULL: usize = 1024;

/// Вершины позиций `0..n` (0 - корень, его вершина - 0) по оценке `score(зависимое, вершина)`
pub(crate) fn heads(n: usize, score: &impl Fn(usize, usize) -> f32) -> Vec<usize> {
    if n > FULL + 1 {
        return long(n, score);
    }
    let mut s = vec![f32::NEG_INFINITY; n * n];
    for d in 1..n {
        for h in 0..n {
            if h != d {
                s[d * n + h] = score(d, h);
            }
        }
    }
    let mut heads: Vec<usize> = (0..n)
        .map(|d| {
            if d == 0 {
                0
            } else {
                argmax(&s[d * n..(d + 1) * n])
            }
        })
        .collect();
    heads[0] = 0;
    if roots(&heads) == 1 && cycle(&heads).is_none() {
        return heads;
    }
    // Логарифмы вероятностей вершин по строке, как у обучения
    let mut logp = vec![vec![f64::NEG_INFINITY; n]; n];
    for d in 1..n {
        let row = &s[d * n..(d + 1) * n];
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        let sum: f64 = row.iter().map(|&x| (x as f64 - max).exp()).sum();
        let log_sum = max + sum.ln();
        for h in 0..n {
            if h != d {
                logp[d][h] = row[h] as f64 - log_sum;
            }
        }
    }
    one_root(&logp)
}

fn argmax(row: &[f32]) -> usize {
    (0..row.len()).fold(0, |best, h| if row[h] > row[best] { h } else { best })
}

fn roots(heads: &[usize]) -> usize {
    heads.iter().skip(1).filter(|&&h| h == 0).count()
}

/// Узлы первого найденного цикла
fn cycle(heads: &[usize]) -> Option<Vec<usize>> {
    let n = heads.len();
    // 0 - не посещен, 1 - на текущем пути, 2 - пути из него к корню проверены
    let mut color = vec![0u8; n];
    for start in 1..n {
        let mut path = Vec::new();
        let mut v = start;
        while v != 0 && color[v] == 0 {
            color[v] = 1;
            path.push(v);
            v = heads[v];
        }
        if v != 0 && color[v] == 1 {
            let at = path.iter().position(|&p| p == v).unwrap_or(0);
            return Some(path[at..].to_vec());
        }
        for p in path {
            color[p] = 2;
        }
    }
    None
}

/// Чу - Лю - Эдмондс с одним корнем: если в лучшем дереве у корня несколько детей, перебираются деревья
/// с каждым из них единственным ребенком корня
fn one_root(scores: &[Vec<f64>]) -> Vec<usize> {
    let n = scores.len();
    let tree = edmonds(scores);
    let candidates: Vec<usize> = (1..n).filter(|&d| tree[d] == 0).collect();
    if candidates.len() == 1 {
        return tree;
    }
    let mut best = (f64::NEG_INFINITY, tree);
    for &root in &candidates {
        let mut s = scores.to_vec();
        for (d, row) in s.iter_mut().enumerate() {
            if d != root {
                row[0] = f64::NEG_INFINITY;
            }
        }
        let candidate = edmonds(&s);
        let total: f64 = (1..n).map(|d| s[d][candidate[d]]).sum();
        if total > best.0 {
            best = (total, candidate);
        }
    }
    best.1
}

/// Дерево максимального веса с корнем 0: `s[d][h]` - оценка вершины `h` у `d`, строка 0 и диагональ - -inf
fn edmonds(s: &[Vec<f64>]) -> Vec<usize> {
    let n = s.len();
    let mut heads: Vec<usize> = (0..n)
        .map(|d| {
            if d == 0 {
                return 0;
            }
            (0..n).fold(0, |best, h| if s[d][h] > s[d][best] { h } else { best })
        })
        .collect();
    heads[0] = 0;
    let Some(cyc) = cycle(&heads) else {
        return heads;
    };
    // Цикл стягивается в узел m; остальные узлы - rest (в нем 0 - первый)
    let mut in_cycle = vec![false; n];
    cyc.iter().for_each(|&v| in_cycle[v] = true);
    let rest: Vec<usize> = (0..n).filter(|&v| !in_cycle[v]).collect();
    let m = rest.len();
    let mut t = vec![vec![f64::NEG_INFINITY; m + 1]; m + 1];
    let mut out_from = vec![0usize; m];
    let mut in_to = vec![0usize; m];
    for (i, &v) in rest.iter().enumerate() {
        for (j, &h) in rest.iter().enumerate() {
            t[i][j] = s[v][h];
        }
        // Лучшая вершина узла v внутри цикла
        let (h, value) = cyc
            .iter()
            .map(|&c| (c, s[v][c]))
            .fold((cyc[0], f64::NEG_INFINITY), max_pair);
        t[i][m] = value;
        out_from[i] = h;
        // Лучший вход в цикл из h = v: замена связи цикла
        let (c, value) = cyc
            .iter()
            .map(|&c| (c, s[c][v] - s[c][heads[c]]))
            .fold((cyc[0], f64::NEG_INFINITY), max_pair);
        t[m][i] = value;
        in_to[i] = c;
    }
    t[0] = vec![f64::NEG_INFINITY; m + 1];
    let inner = edmonds(&t);
    let mut result = heads.clone();
    for i in 1..m {
        result[rest[i]] = if inner[i] == m {
            out_from[i]
        } else {
            rest[inner[i]]
        };
    }
    let h = inner[m];
    result[in_to[h]] = rest[h];
    result
}

fn max_pair(best: (usize, f64), next: (usize, f64)) -> (usize, f64) {
    if next.1 > best.1 { next } else { best }
}

/// Длинное предложение: лучшие вершины без полной матрицы; лишние корни и узлы циклов - к лучшему корню
fn long(n: usize, score: &impl Fn(usize, usize) -> f32) -> Vec<usize> {
    let mut heads = vec![0usize; n];
    let mut root_score = vec![f32::NEG_INFINITY; n];
    for d in 1..n {
        let mut best = (0, f32::NEG_INFINITY);
        for h in 0..n {
            if h == d {
                continue;
            }
            let value = score(d, h);
            if h == 0 {
                root_score[d] = value;
            }
            if value > best.1 {
                best = (h, value);
            }
        }
        heads[d] = best.0;
    }
    let candidates: Vec<usize> = (1..n).filter(|&d| heads[d] == 0).collect();
    let root = if candidates.is_empty() {
        (1..n).fold(1, |best, d| {
            if root_score[d] > root_score[best] {
                d
            } else {
                best
            }
        })
    } else {
        candidates.iter().copied().fold(candidates[0], |best, d| {
            if root_score[d] > root_score[best] {
                d
            } else {
                best
            }
        })
    };
    for (d, head) in heads.iter_mut().enumerate().skip(1) {
        if *head == 0 && d != root {
            *head = root;
        }
    }
    heads[root] = 0;
    while let Some(cyc) = cycle(&heads) {
        heads[cyc[0]] = root;
    }
    heads
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> f32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 40) as f32 / (1u64 << 24) as f32 * 4.0 - 2.0
        }
    }

    fn is_tree(heads: &[usize]) -> bool {
        heads[0] == 0 && roots(heads) == 1 && cycle(heads).is_none()
    }

    /// Лучшее дерево с одним корнем перебором всех вершин
    fn brute(n: usize, s: &[Vec<f64>]) -> f64 {
        let mut best = f64::NEG_INFINITY;
        let mut heads = vec![0usize; n];
        let total = n.pow((n - 1) as u32);
        for code in 0..total {
            let mut c = code;
            for head in heads.iter_mut().skip(1) {
                *head = c % n;
                c /= n;
            }
            if (1..n).any(|d| heads[d] == d) || !is_tree(&heads) {
                continue;
            }
            best = best.max((1..n).map(|d| s[d][heads[d]]).sum());
        }
        best
    }

    #[test]
    fn edmonds_with_one_root_matches_brute_force() {
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        for trial in 0..400 {
            let n = 2 + trial % 6;
            let mut s = vec![vec![f64::NEG_INFINITY; n]; n];
            for (d, row) in s.iter_mut().enumerate().skip(1) {
                for (h, value) in row.iter_mut().enumerate() {
                    if h != d {
                        *value = f64::from(rng.next());
                    }
                }
            }
            let tree = one_root(&s);
            assert!(is_tree(&tree), "{tree:?}");
            let total: f64 = (1..n).map(|d| s[d][tree[d]]).sum();
            assert!((total - brute(n, &s)).abs() < 1e-9, "попытка {trial}");
        }
    }

    #[test]
    fn heads_always_form_tree() {
        let mut rng = Rng(7);
        for n in 2..40 {
            let values: Vec<f32> = (0..n * n).map(|_| rng.next()).collect();
            let score = |d: usize, h: usize| values[d * n + h];
            assert!(is_tree(&heads(n, &score)));
            assert!(is_tree(&long(n, &score)));
        }
    }

    #[test]
    fn argmax_tree_is_kept() {
        // Каждое слово лучше всего цепляется к предыдущему: это уже дерево
        let score = |d: usize, h: usize| if h + 1 == d { 5.0 } else { 0.0 };
        assert_eq!(heads(5, &score), [0, 0, 1, 2, 3]);
    }

    #[test]
    fn long_sentence_without_full_matrix() {
        let n = FULL + 50;
        // Все слова хотят в корень, кроме цикла 5 <-> 6
        let score = |d: usize, h: usize| match (d, h) {
            (5, 6) | (6, 5) => 9.0,
            (_, 0) => 1.0 + d as f32 / n as f32,
            _ => 0.0,
        };
        let tree = heads(n, &score);
        assert!(is_tree(&tree));
        assert_eq!(tree[n - 1], 0);
    }
}
