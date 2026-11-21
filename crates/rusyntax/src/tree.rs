//! Дерево зависимостей по оценкам вершин: у каждого слова одна вершина, корень один, циклов нет.
//!
//! Сначала у каждого слова берется лучшая вершина; если это уже дерево с одним корнем, оно и есть дерево
//! максимального веса. Иначе - Чу - Лю - Эдмондс по логарифмам вероятностей вершин с перебором корня.
//! Работа ограничена (`CONTRACTIONS` стягиваний циклов, `CANDIDATES` кандидатов в корень): сверх этого,
//! как и у предложений длиннее `FULL` слов, лишние корни и узлы циклов подвешиваются к лучшему корню.
//! У предложений длиннее `FULL` слов вершина ищется среди `WINDOW` слов с каждой стороны и корня: время
//! растет линейно, а не квадратично.

/// Длина предложения, до которой строится полная матрица оценок и ищется дерево максимального веса
pub(crate) const FULL: usize = 1024;
/// Сколько циклов стягивает Чу - Лю - Эдмондс (на dev UD больше 6 не бывает)
const CONTRACTIONS: usize = 16;
/// Сколько кандидатов в корень перебирается (на dev UD больше 2 не бывает)
const CANDIDATES: usize = 8;
/// Сколько слов с каждой стороны от зависимого - кандидаты в вершины у предложений длиннее `FULL` слов.
/// В train UD дальше 64 слов - 0,04 % вершин; склейки предложений dev с окном разбираются точнее, чем без него
pub(crate) const WINDOW: usize = 64;

/// Оценка без NaN и бесконечностей: иначе сравнения ломают выбор лучшей вершины
fn finite(value: f32) -> f32 {
    if value.is_nan() {
        f32::MIN
    } else {
        value.clamp(f32::MIN, f32::MAX)
    }
}

/// Вершины позиций `0..n` (0 - корень, его вершина - 0) по оценке `score(зависимое, вершина)` и вероятность
/// каждой вершины - softmax оценок кандидатов в вершины слова, как у обучения (у позиции корня - 1)
pub(crate) fn heads(n: usize, score: &impl Fn(usize, usize) -> f32) -> (Vec<usize>, Vec<f32>) {
    if n > FULL + 1 {
        return long(n, score);
    }
    let mut s = vec![f32::NEG_INFINITY; n * n];
    for d in 1..n {
        for h in 0..n {
            if h != d {
                s[d * n + h] = finite(score(d, h));
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
    let log_sums: Vec<f64> = (0..n)
        .map(|d| {
            if d == 0 {
                0.0
            } else {
                log_sum(&s[d * n..(d + 1) * n])
            }
        })
        .collect();
    if roots(&heads) != 1 || cycle(&heads).is_some() {
        // Логарифмы вероятностей вершин по строке
        let mut logp = vec![vec![f64::NEG_INFINITY; n]; n];
        for d in 1..n {
            for h in 0..n {
                if h != d {
                    logp[d][h] = s[d * n + h] as f64 - log_sums[d];
                }
            }
        }
        heads = one_root(&logp).unwrap_or_else(|| {
            let root_score: Vec<f32> = (0..n).map(|d| s[d * n]).collect();
            repair(heads, &root_score)
        });
    }
    let probs = (0..n)
        .map(|d| {
            if d == 0 {
                1.0
            } else {
                (s[d * n + heads[d]] as f64 - log_sums[d]).exp() as f32
            }
        })
        .collect();
    (heads, probs)
}

/// Вероятность вершины `head` у слова `d` среди тех же кандидатов, что у `heads`: все слова и корень,
/// у предложений длиннее `FULL` слов - окно, корень и `head`; для вершины, назначенной после `heads`
pub(crate) fn head_prob(
    n: usize,
    d: usize,
    head: usize,
    score: &impl Fn(usize, usize) -> f32,
) -> f32 {
    let candidates: Vec<usize> = if n > FULL + 1 {
        let window = d.saturating_sub(WINDOW).max(1)..=(d + WINDOW).min(n - 1);
        let outside = (head != 0 && head.abs_diff(d) > WINDOW).then_some(head);
        std::iter::once(0).chain(window).chain(outside).collect()
    } else {
        (0..n).collect()
    };
    let row: Vec<f32> = (candidates.into_iter())
        .filter(|&h| h != d)
        .map(|h| finite(score(d, h)))
        .collect();
    (finite(score(d, head)) as f64 - log_sum(&row)).exp() as f32
}

/// Логарифм суммы экспонент оценок
fn log_sum(row: &[f32]) -> f64 {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let sum: f64 = row.iter().map(|&x| (x as f64 - max).exp()).sum();
    max + sum.ln()
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
/// с каждым из них единственным ребенком корня. `None` - работа вышла за пределы
fn one_root(scores: &[Vec<f64>]) -> Option<Vec<usize>> {
    let n = scores.len();
    let tree = edmonds(scores.to_vec())?;
    let candidates: Vec<usize> = (1..n).filter(|&d| tree[d] == 0).collect();
    if candidates.len() == 1 {
        return Some(tree);
    }
    if candidates.len() > CANDIDATES {
        return None;
    }
    let mut best = (f64::NEG_INFINITY, tree);
    for &root in &candidates {
        let mut s = scores.to_vec();
        for (d, row) in s.iter_mut().enumerate() {
            if d != root {
                row[0] = f64::NEG_INFINITY;
            }
        }
        let total = |tree: &[usize]| -> f64 { (1..n).map(|d| s[d][tree[d]]).sum() };
        let candidate = edmonds(s.clone())?;
        let value = total(&candidate);
        if value > best.0 {
            best = (value, candidate);
        }
    }
    Some(best.1)
}

/// Стягивание цикла: что нужно, чтобы развернуть вершины графа уровнем ниже
struct Level {
    heads: Vec<usize>,
    rest: Vec<usize>,
    out_from: Vec<usize>,
    in_to: Vec<usize>,
}

/// Дерево максимального веса с корнем 0: `s[d][h]` - оценка вершины `h` у `d`, строка 0 и диагональ - -inf.
/// Без рекурсии: в памяти одна матрица текущего уровня; больше `CONTRACTIONS` стягиваний - `None`
fn edmonds(mut s: Vec<Vec<f64>>) -> Option<Vec<usize>> {
    let mut levels: Vec<Level> = Vec::new();
    let mut result = loop {
        let n = s.len();
        let mut heads: Vec<usize> = (0..n)
            .map(|d| (0..n).fold(0, |best, h| if s[d][h] > s[d][best] { h } else { best }))
            .collect();
        heads[0] = 0;
        let Some(cyc) = cycle(&heads) else {
            break heads;
        };
        if levels.len() == CONTRACTIONS {
            return None;
        }
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
            let (h, value) = (cyc.iter())
                .map(|&c| (c, s[v][c]))
                .fold((cyc[0], f64::NEG_INFINITY), max_pair);
            t[i][m] = value;
            out_from[i] = h;
            // Лучший вход в цикл из v: замена связи цикла
            let (c, value) = (cyc.iter())
                .map(|&c| (c, s[c][v] - s[c][heads[c]]))
                .fold((cyc[0], f64::NEG_INFINITY), max_pair);
            t[m][i] = value;
            in_to[i] = c;
        }
        t[0] = vec![f64::NEG_INFINITY; m + 1];
        levels.push(Level {
            heads,
            rest,
            out_from,
            in_to,
        });
        s = t;
    };
    while let Some(level) = levels.pop() {
        let m = level.rest.len();
        let mut outer = level.heads;
        for i in 1..m {
            outer[level.rest[i]] = if result[i] == m {
                level.out_from[i]
            } else {
                level.rest[result[i]]
            };
        }
        let h = result[m];
        outer[level.in_to[h]] = level.rest[h];
        result = outer;
    }
    Some(result)
}

fn max_pair(best: (usize, f64), next: (usize, f64)) -> (usize, f64) {
    if next.1 > best.1 { next } else { best }
}

/// Длинное предложение: лучшие вершины среди корня и `WINDOW` слов с каждой стороны, затем `repair`;
/// вероятность вершины - среди тех же кандидатов
fn long(n: usize, score: &impl Fn(usize, usize) -> f32) -> (Vec<usize>, Vec<f32>) {
    let mut heads = vec![0usize; n];
    let mut root_score = vec![f32::MIN; n];
    let mut log_sums = vec![0.0f64; n];
    let mut values = Vec::with_capacity(2 * WINDOW + 1);
    for d in 1..n {
        let mut best = (0, f32::NEG_INFINITY);
        let window = d.saturating_sub(WINDOW).max(1)..=(d + WINDOW).min(n - 1);
        values.clear();
        for h in std::iter::once(0).chain(window) {
            if h == d {
                continue;
            }
            let value = finite(score(d, h));
            values.push(value);
            if h == 0 {
                root_score[d] = value;
            }
            if value > best.1 {
                best = (h, value);
            }
        }
        heads[d] = best.0;
        log_sums[d] = log_sum(&values);
    }
    let heads = repair(heads, &root_score);
    let probs = (0..n)
        .map(|d| {
            if d == 0 {
                return 1.0;
            }
            let value = finite(score(d, heads[d])) as f64;
            // Вершину за окном назначила починка: она - еще один кандидат
            let log_sum = if heads[d] == 0 || heads[d].abs_diff(d) <= WINDOW {
                log_sums[d]
            } else {
                let max = log_sums[d].max(value);
                max + ((log_sums[d] - max).exp() + (value - max).exp()).ln()
            };
            (value - log_sum).exp() as f32
        })
        .collect();
    (heads, probs)
}

/// Дерево из лучших вершин: из кандидатов в корень (или из всех слов, если их нет) остается один
/// с лучшей оценкой корня, остальные кандидаты и узлы циклов подвешиваются к нему
fn repair(mut heads: Vec<usize>, root_score: &[f32]) -> Vec<usize> {
    let n = heads.len();
    let candidates: Vec<usize> = (1..n).filter(|&d| heads[d] == 0).collect();
    let pool: Vec<usize> = if candidates.is_empty() {
        (1..n).collect()
    } else {
        candidates
    };
    let root = pool.iter().copied().fold(pool[0], |best, d| {
        if root_score[d] > root_score[best] {
            d
        } else {
            best
        }
    });
    for (d, head) in heads.iter_mut().enumerate().skip(1) {
        if *head == 0 && d != root {
            *head = root;
        }
    }
    heads[root] = 0;
    // Все циклы за один проход: узел, на котором путь замкнулся, подвешивается к корню. Пути с меньших
    // номеров уже проверены и через этот узел не идут, поэтому циклы и порядок те же, что у поиска
    // по одному циклу с начала, а время линейно
    let mut color = vec![0u8; n];
    let mut path = Vec::new();
    for start in 1..n {
        let mut v = start;
        while v != 0 && color[v] == 0 {
            color[v] = 1;
            path.push(v);
            v = heads[v];
        }
        if v != 0 && color[v] == 1 {
            heads[v] = root;
        }
        for p in path.drain(..) {
            color[p] = 2;
        }
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
        for trial in 0..2000 {
            let n = 2 + trial % 6;
            // В половине графов у лучших вершин несколько детей корня: перебираются кандидаты
            let root_bias = if trial % 2 == 0 { 3.0 } else { 0.0 };
            let mut s = vec![vec![f64::NEG_INFINITY; n]; n];
            for (d, row) in s.iter_mut().enumerate().skip(1) {
                for (h, value) in row.iter_mut().enumerate() {
                    if h != d {
                        *value = f64::from(rng.next()) + if h == 0 { root_bias } else { 0.0 };
                    }
                }
            }
            let tree = one_root(&s).expect("малый граф укладывается в пределы");
            assert!(is_tree(&tree), "{tree:?}");
            let total: f64 = (1..n).map(|d| s[d][tree[d]]).sum();
            assert!((total - brute(n, &s)).abs() < 1e-9, "попытка {trial}");
        }
    }

    #[test]
    fn head_probabilities_are_softmax_of_rows() {
        let mut rng = Rng(11);
        for n in [2, 7, 30, FULL + 40] {
            let values: Vec<f32> = (0..n * n.min(200)).map(|_| rng.next()).collect();
            let score =
                |d: usize, h: usize| values[(d * n.min(200) + h % n.min(200)) % values.len()];
            let (tree, probs) = heads(n, &score);
            assert_eq!(probs[0], 1.0);
            for d in 1..n {
                let p = probs[d];
                assert!(p > 0.0 && p <= 1.0, "{d}: {p}");
                if n <= FULL {
                    let row: Vec<f32> = (0..n).filter(|&h| h != d).map(|h| score(d, h)).collect();
                    let sum: f64 = row.iter().map(|&x| f64::from(x).exp()).sum();
                    let expected = f64::from(score(d, tree[d])).exp() / sum;
                    assert!(
                        (f64::from(p) - expected).abs() < 1e-5,
                        "{d}: {p} {expected}"
                    );
                }
            }
        }
    }

    #[test]
    fn head_prob_matches_heads() {
        // Вершина, назначенная после heads, получает вероятность среди тех же кандидатов
        let mut rng = Rng(13);
        for n in [5, 40, FULL + 1, FULL + 300] {
            let width = n.min(97);
            let values: Vec<f32> = (0..n * width).map(|_| rng.next()).collect();
            let score = |d: usize, h: usize| values[d * width + h % width];
            let (tree, probs) = heads(n, &score);
            for d in 1..n {
                let p = head_prob(n, d, tree[d], &score);
                assert!(
                    (p - probs[d]).abs() < 1e-6,
                    "n {n}, слово {d}: {p} против {}",
                    probs[d]
                );
            }
        }
    }

    #[test]
    fn heads_always_form_tree() {
        let mut rng = Rng(7);
        for n in 2..40 {
            let values: Vec<f32> = (0..n * n).map(|_| rng.next()).collect();
            let score = |d: usize, h: usize| values[d * n + h];
            assert!(is_tree(&heads(n, &score).0));
            assert!(is_tree(&long(n, &score).0));
        }
    }

    #[test]
    fn nan_and_infinite_scores_give_tree() {
        // Дерево и вероятности от 0 до 1
        let check = |(tree, probs): (Vec<usize>, Vec<f32>)| {
            is_tree(&tree) && probs.iter().all(|p| (0.0..=1.0).contains(p))
        };
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(check(heads(5, &|_, _| bad)));
            assert!(check(heads(5, &|d, h| if h == 0 { bad } else { d as f32 })));
            assert!(check(long(5, &|_, _| bad)));
        }
    }

    #[test]
    fn degenerate_sentence_is_bounded_and_stack_safe() {
        // Пары соседей хотят друг друга: сотни 2-циклов, у потоков musl стек 128 КиБ
        let n = FULL + 1;
        let score = move |d: usize, h: usize| {
            if h != 0 && d.div_ceil(2) == h.div_ceil(2) {
                10.0
            } else {
                -(d as f32) / n as f32
            }
        };
        let tree = std::thread::Builder::new()
            .stack_size(128 * 1024)
            .spawn(move || heads(n, &score).0)
            .and_then(|thread| {
                thread
                    .join()
                    .map_err(|_| std::io::Error::other("поток упал"))
            })
            .expect("разбор в малом стеке");
        assert!(is_tree(&tree));
    }

    /// Прежняя починка: циклы по одному, каждый раз поиск с начала
    fn repair_one_by_one(mut heads: Vec<usize>, root_score: &[f32]) -> Vec<usize> {
        let root = repair(heads.clone(), root_score)
            .iter()
            .skip(1)
            .position(|&h| h == 0)
            .map_or(1, |at| at + 1);
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

    #[test]
    fn repair_in_one_pass_matches_one_by_one() {
        let mut rng = Rng(11);
        for trial in 0..3000 {
            let n = 2 + trial % 40;
            let heads: Vec<usize> = (0..n)
                .map(|d| {
                    if d == 0 {
                        0
                    } else {
                        ((rng.next() + 2.0) / 4.0 * n as f32) as usize % n
                    }
                })
                .collect();
            let root_score: Vec<f32> = (0..n).map(|_| rng.next()).collect();
            let tree = repair(heads.clone(), &root_score);
            assert!(is_tree(&tree), "попытка {trial}");
            assert_eq!(
                tree,
                repair_one_by_one(heads, &root_score),
                "попытка {trial}"
            );
        }
    }

    #[test]
    fn argmax_tree_is_kept() {
        // Каждое слово лучше всего цепляется к предыдущему: это уже дерево
        let score = |d: usize, h: usize| if h + 1 == d { 5.0 } else { 0.0 };
        let (tree, probs) = heads(5, &score);
        assert_eq!(tree, [0, 0, 1, 2, 3]);
        // У слова 2: e⁵ у вершины 1 и e⁰ у трех других кандидатов
        let e5 = 5f32.exp();
        assert!((probs[2] - e5 / (e5 + 3.0)).abs() < 1e-6);
    }

    #[test]
    fn long_sentence_looks_for_heads_in_window() {
        let n = FULL + 500;
        // Каждое слово лучше всего цепляется к первому слову, но у дальних оно за окном
        let score = |d: usize, h: usize| match h {
            0 => {
                if d == 1 {
                    1.0
                } else {
                    -10.0
                }
            }
            1 => 5.0,
            _ => -(d.abs_diff(h) as f32),
        };
        let tree = heads(n, &score).0;
        assert!(is_tree(&tree));
        assert!((2..=WINDOW + 1).all(|d| tree[d] == 1));
        assert!((WINDOW + 2..n).all(|d| tree[d].abs_diff(d) == 1));
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
        let tree = heads(n, &score).0;
        assert!(is_tree(&tree));
        assert_eq!(tree[n - 1], 0);
    }
}
