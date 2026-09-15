# modules/constants.py
# 複数のJPX手口(参加者別売買)解析スクリプトで共有する定数。

# 日経225先物/オプションの「CTA系フロー」の代理として扱う証券会社名キーワード。
# update_daily_features_v6.py(先物建玉/CTA純建玉)と
# parse_flow_signal.py(日次手口出来高)の両方で参照する。
CTA_BROKER_KEYWORDS = ['ABNクリアリン', 'バークレイズ', 'ソシエテ']
