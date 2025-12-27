"""App name to package name mapping for supported applications."""

APP_PACKAGES: dict[str, str] = {
    # Social & Messaging
    "微信": "com.tencent.mm",
    "QQ": "com.tencent.mobileqq",
    "微博": "com.sina.weibo",
    # E-commerce
    "淘宝": "com.taobao.taobao",
    "京东": "com.jingdong.app.mall",
    "拼多多": "com.xunmeng.pinduoduo",
    "淘宝闪购": "com.taobao.taobao",
    "京东秒送": "com.jingdong.app.mall",
    "闲鱼": "com.taobao.idlefish",
    # Knowledge & Education
    "得到": "com.luojilab.player",
    # Lifestyle & Social
    "小红书": "com.xingin.xhs",
    "豆瓣": "com.douban.frodo",
    "知乎": "com.zhihu.android",
    # Maps & Navigation
    "高德地图": "com.autonavi.minimap",
    "百度地图": "com.baidu.BaiduMap",
    # Food & Services
    "美团": "com.sankuai.meituan",
    "大众点评": "com.dianping.v1",
    "饿了么": "me.ele",
    "肯德基": "com.yek.android.kfc.activitys",
    # Travel
    "携程": "ctrip.android.view",
    "铁路12306": "com.MobileTicket",
    "12306": "com.MobileTicket",
    "去哪儿": "com.Qunar",
    "去哪儿旅行": "com.Qunar",
    "滴滴出行": "com.sdu.didi.psnger",
    # Video & Entertainment
    "bilibili": "tv.danmaku.bili",
    "抖音": "com.ss.android.ugc.aweme",
    "快手": "com.smile.gifmaker",
    "腾讯视频": "com.tencent.qqlive",
    "爱奇艺": "com.qiyi.video",
    "优酷视频": "com.youku.phone",
    "芒果TV": "com.hunantv.imgo.activity",
    "红果短剧": "com.phoenix.read",
    # Music & Audio
    "网易云音乐": "com.netease.cloudmusic",
    "QQ音乐": "com.tencent.qqmusic",
    "汽水音乐": "com.luna.music",
    "喜马拉雅": "com.ximalaya.ting.android",
    # Reading
    "番茄小说": "com.dragon.read",
    "番茄免费小说": "com.dragon.read",
    "七猫免费小说": "com.kmxs.reader",
    # Productivity
    "飞书": "com.ss.android.lark",
    "QQ邮箱": "com.tencent.androidqqmail",
    # AI & Tools
    "豆包": "com.larus.nova",
    # Health & Fitness
    "keep": "com.gotokeep.keep",
    "美柚": "com.lingan.seeyou",
    # News & Information
    "腾讯新闻": "com.tencent.news",
    "今日头条": "com.ss.android.article.news",
    # Real Estate
    "贝壳找房": "com.lianjia.beike",
    "安居客": "com.anjuke.android.app",
    # Finance
    "同花顺": "com.hexin.plat.android",
}


def get_package_name(app_name: str) -> str | None:
    """
    Get the package name for an app.

    Args:
        app_name: The display name of the app.

    Returns:
        The Android package name, or None if not found.
    """
    return APP_PACKAGES.get(app_name)


def get_app_name(package_name: str) -> str | None:
    """
    Get the app name from a package name.

    Args:
        package_name: The Android package name.

    Returns:
        The display name of the app, or None if not found.
    """
    for name, package in APP_PACKAGES.items():
        if package == package_name:
            return name
    return None


def list_supported_apps() -> list[str]:
    """
    Get a list of all supported app names.

    Returns:
        List of app names.
    """
    return list(APP_PACKAGES.keys())