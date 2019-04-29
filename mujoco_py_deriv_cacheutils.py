from functools import wraps

def cached_method(method, cache_name='_cache'):
    default = object()
    @wraps(method)
    def wrapper(self):
        name = method.__name__
        _cache = getattr(self, cache_name, dict())
        cache = (_cache
                 if isinstance(_cache, dict)
                 else vars(_cache))
        setattr(self, cache_name, _cache)
        val = cache.get(name, default)
        if val is default:
            val = method(self)
            cache[name] = val

        assert val is not default
        return val
    return wrapper


def cached_property(method, **kw):
    return property(cached_method(method, **kw))

