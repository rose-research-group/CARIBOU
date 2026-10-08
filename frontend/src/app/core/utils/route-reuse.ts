import { Injectable } from '@angular/core';
import { ActivatedRouteSnapshot, BaseRouteReuseStrategy } from '@angular/router';

/**
 * Angular reuses a component when only its route params change, but the
 * session page reads its id once and provides its own SessionStore. Moving
 * between sessions in place (a branch lane, the link back to a branch's
 * parent) must therefore build a fresh page: `session/:id` is reused only
 * when its id is unchanged. Query-param and fragment changes (`?view=`,
 * `#block:`) still reuse it. Every other route keeps Angular's default.
 */
const SESSION_ROUTE_PATH = 'session/:id';

@Injectable()
export class ParamAwareRouteReuseStrategy extends BaseRouteReuseStrategy {
  override shouldReuseRoute(future: ActivatedRouteSnapshot, curr: ActivatedRouteSnapshot): boolean {
    if (future.routeConfig?.path !== SESSION_ROUTE_PATH) return super.shouldReuseRoute(future, curr);
    return future.routeConfig === curr.routeConfig && sameParams(future.params, curr.params);
  }
}

function sameParams(a: Record<string, unknown>, b: Record<string, unknown>): boolean {
  const keys = Object.keys(a);
  return keys.length === Object.keys(b).length && keys.every(k => a[k] === b[k]);
}
