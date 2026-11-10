use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use sudachi::analysis::Mode;
use sudachi::analysis::stateful_tokenizer::StatefulTokenizer;
use sudachi::config::ConfigBuilder;
use sudachi::dic::build::DictBuilder;
use sudachi::dic::dictionary::JapaneseDictionary;
use sudachi::dic::storage::{Storage, SudachiDicData};
use sudachi::prelude::MorphemeList;
use tantivy::tokenizer::{ MAX_TOKEN_LEN, RawTokenizer, Token, TokenStream, Tokenizer, TokenizerManager };

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "UPPERCASE")]
pub enum SplitMode {
    #[serde(alias = "a", alias = "short")]
    A,
    #[serde(alias = "b", alias = "middle")]
    B,
    #[default]
    #[serde(alias = "c", alias = "long")]
    C,
}

impl From<SplitMode> for Mode {
    fn from(m: SplitMode) -> Self {
        match m {
            SplitMode::A => Mode::A,
            SplitMode::B => Mode::B,
            SplitMode::C => Mode::C,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum TokenForm {
    #[default]
    #[serde(alias = "normalized_form")]
    Normalized,
    #[serde(alias = "dictionary_form", alias = "lemma")]
    Dictionary,
    #[serde(alias = "surface_form")]
    Surface,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SudachiTokenizerConfig {
    #[serde(default)]
    pub mode: SplitMode,
    #[serde(default)]
    pub form: TokenForm,
    #[serde(default = "default_stop_tags")]
    pub stop_tags: Vec<String>,
    #[serde(default)]
    pub stop_words: HashSet<String>,
    #[serde(default = "default_true")]
    pub lowercase: bool,
}

fn default_stop_tags() -> Vec<String> {
    vec!["補助記号".to_string(), "空白".to_string()]
}

fn default_true() -> bool {
    true
}

impl Default for SudachiTokenizerConfig {
    fn default() -> Self {
        Self {
            mode: SplitMode::C,
            form: TokenForm::Normalized,
            stop_tags: default_stop_tags(),
            stop_words: HashSet::new(),
            lowercase: true,
        }
    }
}

impl SudachiTokenizerConfig {
    pub fn with_mode(mode: SplitMode) -> Self {
        Self {
            mode,
            ..Self::default()
        }
    }

    fn should_filter_pos(&self, pos: &[String]) -> bool {
        if self.stop_tags.is_empty() {
            return false;
        }
        let joined = pos.join(",");
        self.stop_tags.iter().any(|tag| {
            pos.first().is_some_and(|first| first == tag) || joined.starts_with(tag)
        })
    }
}

#[derive(Clone)]
pub struct SudachiTokenizer {
    dict: Arc<JapaneseDictionary>,
    config: Arc<SudachiTokenizerConfig>,
    tokens: Vec<Token>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AnalyzeToken {
    pub token: String,
    pub start_offset: usize,
    pub end_offset: usize,
    #[serde(rename = "type")]
    pub pos: String,
    pub position: usize,
    pub reading: String,
    pub normalized_form: String,
    pub dictionary_form: String,
}

impl SudachiTokenizer {
    pub fn new(dict: Arc<JapaneseDictionary>, config: SudachiTokenizerConfig) -> Self {
        Self {
            dict,
            config: Arc::new(config),
            tokens: Vec::new(),
        }
    }

    pub fn analyze(&self, text: &str) -> Vec<AnalyzeToken> {
        let mut stateful = StatefulTokenizer::new(self.dict.as_ref(), self.config.mode.into());
        stateful.reset().push_str(text);
        if stateful.do_tokenize().is_err() {
            return Vec::new();
        }
        let mut morphemes = MorphemeList::empty(self.dict.as_ref());
        if morphemes.collect_results(&mut stateful).is_err() {
            return Vec::new();
        }

        let mut out = Vec::with_capacity(morphemes.len());
        let mut pos_idx = 0;

        for m in morphemes.iter() {
            let pos = m.part_of_speech();
            if self.config.should_filter_pos(pos) {
                continue;
            }
            let surface = m.surface();
            let raw_text = match self.config.form {
                TokenForm::Normalized => m.normalized_form(),
                TokenForm::Dictionary => m.dictionary_form(),
                TokenForm::Surface => surface.as_ref(),
            };
            let trimmed = raw_text.trim();
            if trimmed.is_empty() || trimmed.len() >= MAX_TOKEN_LEN {
                continue;
            }
            let token_str = if self.config.lowercase {
                trimmed.to_lowercase()
            } else {
                trimmed.to_string()
            };
            if self.config.stop_words.contains(&token_str) {
                continue;
            }
            out.push(AnalyzeToken {
                token: token_str,
                start_offset: m.begin(),
                end_offset: m.end(),
                pos: pos.first().cloned().unwrap_or_default(),
                position: pos_idx,
                reading: m.reading_form().to_string(),
                normalized_form: m.normalized_form().to_string(),
                dictionary_form: m.dictionary_form().to_string(),
            });
            pos_idx += 1;
        }
        out
    }
}

pub struct SudachiTokenStream<'a> {
    tokens: &'a [Token],
    index: usize,
}

impl Tokenizer for SudachiTokenizer {
    type TokenStream<'a> = SudachiTokenStream<'a>;

    fn token_stream<'a>(&'a mut self, text: &'a str) -> Self::TokenStream<'a> {
        self.tokens.clear();

        let mut stateful = StatefulTokenizer::new(self.dict.as_ref(), self.config.mode.into());
        stateful.reset().push_str(text);
        if stateful.do_tokenize().is_ok() {
            let mut morphemes = MorphemeList::empty(self.dict.as_ref());
            if morphemes.collect_results(&mut stateful).is_ok() {
                let mut position = 0;
                for m in morphemes.iter() {
                    let pos = m.part_of_speech();
                    if self.config.should_filter_pos(pos) {
                        continue;
                    }
                    let surface = m.surface();
                    let raw_text = match self.config.form {
                        TokenForm::Normalized => m.normalized_form(),
                        TokenForm::Dictionary => m.dictionary_form(),
                        TokenForm::Surface => surface.as_ref(),
                    };
                    let trimmed = raw_text.trim();
                    if trimmed.is_empty() || trimmed.len() >= MAX_TOKEN_LEN {
                        continue;
                    }
                    let token_text = if self.config.lowercase {
                        trimmed.to_lowercase()
                    } else {
                        trimmed.to_string()
                    };
                    if self.config.stop_words.contains(&token_text) {
                        continue;
                    }
                    self.tokens.push(Token {
                        offset_from: m.begin(),
                        offset_to: m.end(),
                        position,
                        text: token_text,
                        position_length: 1,
                    });
                    position += 1;
                }
            }
        }

        SudachiTokenStream {
            tokens: &self.tokens,
            index: 0,
        }
    }
}

impl TokenStream for SudachiTokenStream<'_> {
    fn advance(&mut self) -> bool {
        if self.index < self.tokens.len() {
            self.index += 1;
            true
        } else {
            false
        }
    }

    fn token(&self) -> &Token {
        &self.tokens[self.index - 1]
    }

    fn token_mut(&mut self) -> &mut Token {
        unimplemented!("SudachiTokenStream does not support in-place mutation")
    }
}

/// Register built-in Japanese and keyword tokenizers on a Tantivy `TokenizerManager`.
pub fn register_default_tokenizers(
    manager: &TokenizerManager,
    dict: &Arc<JapaneseDictionary>,
) {
    manager.register(
        "sudachi",
        SudachiTokenizer::new(dict.clone(), SudachiTokenizerConfig::with_mode(SplitMode::C)),
    );
    manager.register(
        "sudachi_a",
        SudachiTokenizer::new(dict.clone(), SudachiTokenizerConfig::with_mode(SplitMode::A)),
    );
    manager.register(
        "sudachi_b",
        SudachiTokenizer::new(dict.clone(), SudachiTokenizerConfig::with_mode(SplitMode::B)),
    );
    manager.register(
        "sudachi_c",
        SudachiTokenizer::new(dict.clone(), SudachiTokenizerConfig::with_mode(SplitMode::C)),
    );
    manager.register(
        "sudachi_search",
        SudachiTokenizer::new(
            dict.clone(),
            SudachiTokenizerConfig {
                mode: SplitMode::A,
                form: TokenForm::Normalized,
                stop_tags: vec![
                    "補助記号".to_string(),
                    "空白".to_string(),
                    "助詞".to_string(),
                    "助動詞".to_string(),
                ],
                stop_words: HashSet::new(),
                lowercase: true,
            },
        ),
    );
    manager.register("keyword", RawTokenizer::default());
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DiscoveredDictionaries {
    pub system_dict: Option<PathBuf>,
    pub user_dicts: Vec<PathBuf>,
}

/// Discovers Sudachi dictionary files (`.dic`) within a directory.
///
/// Identification rules:
/// - A file named `system.dic`, `system_core.dic`, `system_full.dic`, `system_small.dic`,
///   or matching `system*.dic` (case-insensitive) is identified as the system dictionary.
///   Priority order: `system.dic` > `system_core/full/small.dic` > `system*.dic`.
/// - All other `.dic` files are treated as user dictionaries and returned in alphabetical order.
pub fn discover_dictionaries(dir: &Path) -> Result<DiscoveredDictionaries, String> {
    if !dir.is_dir() {
        return Ok(DiscoveredDictionaries::default());
    }

    let entries = std::fs::read_dir(dir)
        .map_err(|e| format!("failed to read dictionary directory {}: {e}", dir.display()))?;

    let mut system_candidate: Option<(u8, PathBuf)> = None;
    let mut other_dics: Vec<PathBuf> = Vec::new();

    for entry in entries {
        let entry = entry.map_err(|e| format!("failed to read directory entry in {}: {e}", dir.display()))?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }

        let is_dic = path.extension().is_some_and(|ext| ext.eq_ignore_ascii_case("dic"));
        if !is_dic {
            continue;
        }

        let file_name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default()
            .to_ascii_lowercase();

        let priority = if file_name == "system.dic" {
            Some(0)
        } else if file_name == "system_core.dic"
            || file_name == "system_full.dic"
            || file_name == "system_small.dic"
        {
            Some(1)
        } else if file_name.starts_with("system") {
            Some(2)
        } else {
            None
        };

        if let Some(prio) = priority {
            match &system_candidate {
                None => system_candidate = Some((prio, path)),
                Some((cur_prio, cur_path)) => {
                    if prio < *cur_prio || (prio == *cur_prio && path < *cur_path) {
                        other_dics.push(cur_path.clone());
                        system_candidate = Some((prio, path));
                    } else {
                        other_dics.push(path);
                    }
                }
            }
        } else {
            other_dics.push(path);
        }
    }

    other_dics.sort();

    Ok(DiscoveredDictionaries {
        system_dict: system_candidate.map(|(_, p)| p),
        user_dicts: other_dics,
    })
}

/// Loads a Sudachi dictionary with optional user dictionaries.
///
/// Priority resolution:
/// 1. If `dict_dir` is specified, dictionaries are discovered in that directory.
///    - If `explicit_sys_dict` is also provided, it overrides any system dictionary found in `dict_dir`.
/// 2. If `dict_dir` is not specified, but `explicit_sys_dict` is given, that system dictionary is used,
///    and its parent directory is scanned for any accompanying user dictionaries.
/// 3. If neither is specified, `./resources` is checked as a default dictionary directory.
/// 4. If no valid system dictionary is found anywhere, falls back to the built-in minimal dictionary.
pub fn load_dictionary_from_dir(
    dict_dir: Option<&Path>,
    explicit_sys_dict: Option<&Path>,
) -> Result<Arc<JapaneseDictionary>, String> {
    let mut resolved_system: Option<PathBuf> = None;
    let mut resolved_users: Vec<PathBuf> = Vec::new();

    // 1. Explicit system dictionary argument takes precedence if it exists
    if let Some(sys_path) = explicit_sys_dict {
        if sys_path.exists() {
            resolved_system = Some(sys_path.to_path_buf());
        } else {
            tracing::warn!(
                "Specified system dictionary not found at {}",
                sys_path.display()
            );
        }
    }

    // 2. Determine directory to scan for dictionaries
    let target_dir = dict_dir
        .map(Path::to_path_buf)
        .or_else(|| {
            // If explicit system dict was given, check its parent directory for user dicts
            explicit_sys_dict.and_then(|p| p.parent().map(Path::to_path_buf))
        })
        .or_else(|| {
            let default_dir = Path::new("./resources");
            if default_dir.is_dir() {
                Some(default_dir.to_path_buf())
            } else {
                None
            }
        });

    if let Some(ref dir) = target_dir {
        if dir.is_dir() {
            let discovered = discover_dictionaries(&dir)?;
            if resolved_system.is_none() {
                resolved_system = discovered.system_dict;
            }
            for u in discovered.user_dicts {
                if Some(&u) != resolved_system.as_ref() {
                    resolved_users.push(u);
                }
            }
        } else {
            tracing::warn!("Dictionary directory does not exist: {}", dir.display());
        }
    }

    // 3. If system dictionary is resolved, build Sudachi config and load via mmap
    if let Some(sys_path) = resolved_system {
        let abs_sys = std::fs::canonicalize(&sys_path)
            .map_err(|e| format!("failed to canonicalize system dict {}: {e}", sys_path.display()))?;

        let mut user_paths_canonical = Vec::new();
        for u in resolved_users {
            let abs_u = std::fs::canonicalize(&u)
                .map_err(|e| format!("failed to canonicalize user dict {}: {e}", u.display()))?;
            user_paths_canonical.push(abs_u);
        }

        // Check if there is a custom sudachi.json in the target directory
        let custom_cfg_file = target_dir
            .as_ref()
            .map(|d| d.join("sudachi.json"))
            .filter(|p| p.is_file());

        let mut builder = match custom_cfg_file {
            Some(ref cfg_path) => ConfigBuilder::from_file(cfg_path)
                .map_err(|e| format!("failed to load {}: {e}", cfg_path.display()))?,
            None => ConfigBuilder::from_opt_file(None)
                .map_err(|e| format!("failed to create Sudachi config: {e}"))?,
        }
        .push_embedded()
        .system_dict(abs_sys.clone());

        for u in &user_paths_canonical {
            builder = builder.user_dict(u.clone());
        }

        let cfg = builder.build();
        let dict = match JapaneseDictionary::from_cfg(&cfg) {
            Ok(d) => d,
            Err(e) if custom_cfg_file.is_none() => {
                // If default configuration failed (e.g. system dictionary has small connection matrix
                // that cannot satisfy default MeCabOovPlugin), retry with lightweight OOV configuration.
                let mut retry_builder = ConfigBuilder::from_bytes(FALLBACK_CFG_JSON.as_bytes())
                    .map_err(|e2| e2.to_string())?
                    .push_embedded()
                    .system_dict(abs_sys.clone());
                for u in &user_paths_canonical {
                    retry_builder = retry_builder.user_dict(u.clone());
                }
                JapaneseDictionary::from_cfg(&retry_builder.build())
                    .map_err(|_| format!("failed to initialize JapaneseDictionary: {e}"))?
            }
            Err(e) => return Err(format!("failed to initialize JapaneseDictionary: {e}")),
        };

        if user_paths_canonical.is_empty() {
            tracing::info!(
                "Loaded Sudachi dictionary via mmap (system: {})",
                abs_sys.display()
            );
        } else {
            tracing::info!(
                "Loaded Sudachi dictionary via mmap (system: {}, {} user dict(s): {:?})",
                abs_sys.display(),
                user_paths_canonical.len(),
                user_paths_canonical
                    .iter()
                    .map(|p| p.file_name().unwrap_or_default().to_string_lossy())
                    .collect::<Vec<_>>()
            );
        }

        return Ok(Arc::new(dict));
    }

    tracing::warn!(
        "No Sudachi system dictionary found (set TANPOPO_SUDACHI_DICT_DIR, TANPOPO_DICT_PATH, or place in ./resources/); using minimal built-in dictionary"
    );
    build_fallback_dictionary()
}

/// Backward compatibility: loads dictionary with an optional system dictionary path.
pub fn load_dictionary(dict_path: Option<&Path>) -> Result<Arc<JapaneseDictionary>, String> {
    load_dictionary_from_dir(None, dict_path)
}

/// Builds minimal system dictionary binary bytes containing core Japanese vocabulary and OOV rules.
pub fn build_fallback_system_dic_bytes() -> Result<Vec<u8>, String> {
    let mut builder = DictBuilder::new_system();
    let mut matrix = String::from("10 10\n");
    for l in 0..10 {
        for r in 0..10 {
            matrix.push_str(&format!("{l} {r} 100\n"));
        }
    }
    builder
        .read_conn(matrix.as_bytes())
        .map_err(|e| format!("conn build error: {e}"))?;

    // Minimal lexicon covering split modes A/B/C, normalization, POS filtering, and OOV POS tags
    let lexicon_csv = concat!(
        // 0: 東京, 1: 都, 2: 東京都
        "東京,1,1,3000,東京,名詞,固有名詞,地名,一般,*,*,トウキョウ,東京,*,A,*,*,*,*\n",
        "都,1,1,3500,都,名詞,普通名詞,一般,*,*,*,ト,都,*,A,*,*,*,*\n",
        "東京都,1,1,2800,東京都,名詞,固有名詞,地名,一般,*,*,トウキョウト,東京都,*,C,0/1,*,0/1,*\n",
        // 3: 京都, 4: 大阪, 5: 関西, 6: 国際, 7: 空港, 8: 国際空港, 9: 関西国際空港
        "京都,1,1,3000,京都,名詞,固有名詞,地名,一般,*,*,キョウト,京都,*,A,*,*,*,*\n",
        "大阪,1,1,3000,大阪,名詞,固有名詞,地名,一般,*,*,オオサカ,大阪,*,A,*,*,*,*\n",
        "関西,1,1,3000,関西,名詞,固有名詞,地名,一般,*,*,カンサイ,関西,*,A,*,*,*,*\n",
        "国際,1,1,3000,国際,名詞,普通名詞,一般,*,*,*,コクサイ,国際,*,A,*,*,*,*\n",
        "空港,1,1,3000,空港,名詞,普通名詞,一般,*,*,*,クウコウ,空港,*,A,*,*,*,*\n",
        "国際空港,1,1,2900,国際空港,名詞,普通名詞,一般,*,*,*,コクサイクウコウ,国際空港,*,B,6/7,*,6/7,*\n",
        "関西国際空港,1,1,2500,関西国際空港,名詞,固有名詞,一般,*,*,*,カンサイコクサイクウコウ,関西国際空港,*,C,5/6/7,5/8,5/6/7,*\n",
        // 10: 全文, 11: 検索, 12: エンジン, 13: 全文検索, 14: 全文検索エンジン
        "全文,1,1,3000,全文,名詞,普通名詞,一般,*,*,*,ゼンブン,全文,*,A,*,*,*,*\n",
        "検索,1,1,3000,検索,名詞,普通名詞,サ変可能,*,*,*,ケンサク,検索,*,A,*,*,*,*\n",
        "エンジン,1,1,3000,エンジン,名詞,普通名詞,一般,*,*,*,エンジン,エンジン,*,A,*,*,*,*\n",
        "全文検索,1,1,2800,全文検索,名詞,普通名詞,一般,*,*,*,ゼンブンケンサク,全文検索,*,B,10/11,*,10/11,*\n",
        "全文検索エンジン,1,1,2400,全文検索エンジン,名詞,普通名詞,一般,*,*,*,ゼンブンケンサクエンジン,全文検索エンジン,*,C,10/11/12,13/12,10/11/12,*\n",
        // 15: 形態素, 16: 解析, 17: 形態素解析
        "形態素,1,1,3000,形態素,名詞,普通名詞,一般,*,*,*,ケイタイソ,形態素,*,A,*,*,*,*\n",
        "解析,1,1,3000,解析,名詞,普通名詞,サ変可能,*,*,*,カイセキ,解析,*,A,*,*,*,*\n",
        "形態素解析,1,1,2600,形態素解析,名詞,普通名詞,一般,*,*,*,ケイタイソカイセキ,形態素解析,*,C,15/16,*,15/16,*\n",
        // General words & spelling normalization examples
        "日本語,1,1,3000,日本語,名詞,普通名詞,一般,*,*,*,ニホンゴ,日本語,*,A,*,*,*,*\n",
        "省メモリ,1,1,3000,省メモリ,名詞,普通名詞,一般,*,*,*,ショウメモリ,省メモリ,*,A,*,*,*,*\n",
        "高速,1,1,3000,高速,形状詞,一般,*,*,*,*,コウソク,高速,*,A,*,*,*,*\n",
        "案内,1,1,3000,案内,名詞,普通名詞,サ変可能,*,*,*,アンナイ,案内,*,A,*,*,*,*\n",
        "方法,1,1,3000,方法,名詞,普通名詞,一般,*,*,*,ホウホウ,方法,*,A,*,*,*,*\n",
        "観察,1,1,3000,観察,名詞,普通名詞,サ変可能,*,*,*,カンサツ,観察,*,A,*,*,*,*\n",
        "記録,1,1,3000,記録,名詞,普通名詞,サ変可能,*,*,*,キロク,記録,*,A,*,*,*,*\n",
        "実現,1,1,3000,実現,名詞,普通名詞,サ変可能,*,*,*,ジツゲン,実現,*,A,*,*,*,*\n",
        "たんぽぽ,1,1,3000,たんぽぽ,名詞,普通名詞,一般,*,*,*,タンポポ,タンポポ,*,A,*,*,*,*\n",
        "タンポポ,1,1,3000,タンポポ,名詞,普通名詞,一般,*,*,*,タンポポ,タンポポ,*,A,*,*,*,*\n",
        "蒲公英,1,1,3000,蒲公英,名詞,普通名詞,一般,*,*,*,タンポポ,タンポポ,*,A,*,*,*,*\n",
        "打込む,1,1,3000,打込む,動詞,一般,*,*,五段-マ行,終止形-一般,ウチコム,打ち込む,*,A,*,*,*,*\n",
        "打ち込む,1,1,3000,打ち込む,動詞,一般,*,*,五段-マ行,終止形-一般,ウチコム,打ち込む,*,A,*,*,*,*\n",
        "醤油,1,1,3000,醤油,名詞,普通名詞,一般,*,*,*,ショウユ,醤油,*,A,*,*,*,*\n",
        "しょうゆ,1,1,3000,しょうゆ,名詞,普通名詞,一般,*,*,*,ショウユ,醤油,*,A,*,*,*,*\n",
        "いく,1,1,3000,いく,動詞,非自立可能,*,*,五段-カ行,終止形-一般,イク,行く,*,A,*,*,*,*\n",
        "行く,1,1,3000,行く,動詞,非自立可能,*,*,五段-カ行,終止形-一般,イク,行く,*,A,*,*,*,*\n",
        // Particles, auxiliary verbs, symbols, and required OOV POS entries
        "に,2,2,3000,に,助詞,格助詞,*,*,*,*,ニ,に,*,A,*,*,*,*\n",
        "の,2,2,3000,の,助詞,格助詞,*,*,*,*,ノ,の,*,A,*,*,*,*\n",
        "を,2,2,3000,を,助詞,格助詞,*,*,*,*,ヲ,を,*,A,*,*,*,*\n",
        "は,2,2,3000,は,助詞,係助詞,*,*,*,*,ハ,は,*,A,*,*,*,*\n",
        "が,2,2,3000,が,助詞,格助詞,*,*,*,*,ガ,が,*,A,*,*,*,*\n",
        "で,2,2,3000,で,助詞,格助詞,*,*,*,*,デ,で,*,A,*,*,*,*\n",
        "と,2,2,3000,と,助詞,格助詞,*,*,*,*,ト,と,*,A,*,*,*,*\n",
        "から,2,2,3000,から,助詞,格助詞,*,*,*,*,カラ,から,*,A,*,*,*,*\n",
        "な,3,3,3000,な,助動詞,*,*,*,助動詞-ダ,連体形-一般,ナ,だ,*,A,*,*,*,*\n",
        "し,1,1,3000,し,動詞,非自立可能,*,*,サ行変格,連用形-一般,シ,する,*,A,*,*,*,*\n",
        "ます,3,3,3000,ます,助動詞,*,*,*,助動詞-マス,終止形-一般,マス,ます,*,A,*,*,*,*\n",
        "です,3,3,3000,です,助動詞,*,*,*,助動詞-デス,終止形-一般,デス,です,*,A,*,*,*,*\n",
        "、,0,0,4000,、,補助記号,読点,*,*,*,*,、,、,*,A,*,*,*,*\n",
        "。,0,0,4000,。,補助記号,句点,*,*,*,*,。,。,*,A,*,*,*,*\n",
        "　,0,0,4000,　,空白,*,*,*,*,*,　,　,*,A,*,*,*,*\n",
        "x,8,8,-20000,X,補助記号,一般,*,*,*,*,X,X,*,A,*,*,*,*\n",
        "x,8,8,-20000,X,感動詞,一般,*,*,*,*,X,X,*,A,*,*,*,*\n",
        "x,8,8,-20000,X,名詞,固有名詞,人名,一般,*,*,X,X,*,A,*,*,*,*\n",
        "x,8,8,-20000,X,名詞,数詞,*,*,*,*,X,X,*,A,*,*,*,*\n"
    );
    builder
        .read_lexicon(lexicon_csv.as_bytes())
        .map_err(|e| format!("lexicon build error: {e}"))?;
    builder
        .resolve()
        .map_err(|e| format!("resolve error: {e}"))?;

    let mut compiled = Vec::new();
    builder
        .compile(&mut compiled)
        .map_err(|e| format!("compile error: {e}"))?;
    Ok(compiled)
}

const FALLBACK_CFG_JSON: &str = r#"{
    "characterDefinitionFile": "char.def",
    "inputTextPlugin": [
        { "class": "com.worksap.nlp.sudachi.DefaultInputTextPlugin" }
    ],
    "oovProviderPlugin": [
        {
            "class": "com.worksap.nlp.sudachi.SimpleOovPlugin",
            "oovPOS": [ "名詞", "普通名詞", "一般", "*", "*", "*" ],
            "leftId": 8,
            "rightId": 8,
            "cost": 6000
        }
    ],
    "pathRewritePlugin": [
        { "class": "com.worksap.nlp.sudachi.JoinNumericPlugin", "enableNormalize": true },
        {
            "class": "com.worksap.nlp.sudachi.JoinKatakanaOovPlugin",
            "oovPOS": [ "名詞", "普通名詞", "一般", "*", "*", "*" ],
            "minLength": 3
        }
    ]
}"#;

/// Builds a lightweight in-memory Sudachi dictionary containing core Japanese vocabulary and OOV rules
/// for zero-dependency testing and quick start.
pub fn build_fallback_dictionary() -> Result<Arc<JapaneseDictionary>, String> {
    let compiled = build_fallback_system_dic_bytes()?;
    let cfg = ConfigBuilder::from_bytes(FALLBACK_CFG_JSON.as_bytes())
        .map_err(|e| format!("config error: {e}"))?
        .push_embedded()
        .build();
    let storage = SudachiDicData::new(Storage::Owned(compiled));
    let dict = JapaneseDictionary::from_cfg_storage(&cfg, storage)
        .map_err(|e| format!("dictionary init error: {e}"))?;
    Ok(Arc::new(dict))
}
