import tensorflow as tf

from hesp.util.hyperbolic_nn import riemannian_gradient_c, tf_exp_map_x, tf_sqnorm, PROJ_EPS, EPS


def cross_correlate(inputs, filters):
    # performs 1x1 convolution of filters over inputs
    return tf.nn.conv2d(inputs, filter=filters, strides=[1, 1, 1, 1], padding="SAME")


def get_hyp_update_ops(var, grads, curvature, learning_rate, burnin=1):
    hyp_train_ops = []
    for i in range(len(var)):
        hyp_train_ops.append(RSGD_op(var[i], grads[i], curvature=curvature, learning_rate=learning_rate, burnin=burnin))
    return tf.group(*hyp_train_ops)


def RSGD_op(v, g, curvature, learning_rate, burnin=1.):
    riemannian_scaling_factor = riemannian_gradient_c(v, c=curvature)
    rescaled_gradient = riemannian_scaling_factor * g
    hyp_update = -(learning_rate * burnin * rescaled_gradient)

    # hyp_updated = tf_project_hyp_vecs(v+hyp_update,curvature) # via retraction
    hyp_updated = tf_exp_map_x(v, hyp_update, c=curvature)
    return tf.assign(v, hyp_updated)


def euc_mlr(inputs, P_mlr, A_mlr):
    with tf.variable_scope("euc_mlr"):
        A_kernel = tf.transpose(A_mlr)[None, None, :, :]
        xdota = cross_correlate(inputs, filters=A_kernel)
        pdota = tf.reduce_sum(-P_mlr * A_mlr, axis=1)[None, None, None, :]
        return pdota + xdota


def hyp_mlr(inputs, c, P_mlr, A_mlr):
    with tf.variable_scope("hyp_mlr"):
        xx = tf_sqnorm(inputs)  # sh B,H,W,1
        pp = tf_sqnorm(-P_mlr, keepdims=False, axis=1)  # sh [ch]
        pp = tf.debugging.check_numerics(pp, 'pp nan')
        # 1x1 conv.
        # | -p * x |^2 p has shape ncls, D, need in-out shape for filter: D,ncls
        P_kernel = tf.transpose(-P_mlr, [1, 0])[None, None, :, :]
        px = cross_correlate(inputs, filters=P_kernel)  # sh B,H,W,ch

        # c^2 * | X|^2 * |-P|^2
        sqsq = tf.multiply(c * xx, c * pp[None, None, None, :])  # sh B,H,W,ch

        # Weight operations
        A_norm = tf.norm(A_mlr, axis=1)  # [ncls,1]
        normed_A = tf.nn.l2_normalize(A_mlr, axis=1)  # impl does this
        A_kernel = tf.transpose(normed_A)[None, None, :, :]

        # rewrite mob add as alpha * p + beta * x
        # where alpha = A/D
        A = 1 + tf.add(2 * c * px, c * xx)  # sh B,H,W,ch
        A = tf.debugging.check_numerics(A, 'B nan')
        # tf.summary.histogram('A',A)
        B = 1 - c * pp  # sh ch
        B = tf.debugging.check_numerics(B, 'B nan')
        # tf.summary.histogram('B',B)
        D = 1 + tf.add(2 * c * px, sqsq)  # sh B,H,W,ch
        D = tf.maximum(D, EPS)
        D = tf.debugging.check_numerics(D, 'D nan')
        # tf.summary.histogram('D',D)
        # calculate mobadd norm indepently from mob add
        # if mob_add = alpha * p + beta * x, then
        #  |mob_add|^2 = theta**2 * |p|^2 + gamma^2 * |x|^2 + 2*theta*gamma*|px|
        # theta = A/D, gamma = B/D
        alpha = A / D  # B,H,W,ch
        alpha = tf.debugging.check_numerics(alpha, 'alpha nan')
        # tf.summary.histogram('alpha',alpha)
        beta = B[None, None, None, :] / D  # B,H,W,ch
        beta = tf.debugging.check_numerics(beta, 'beta nan')
        # tf.summary.histogram('beta',beta)
        # calculate mobius addition norm independently
        mobaddnorm = (
                (alpha ** 2 * pp[None, None, None, :])
                + (beta ** 2 * xx)
                + (2 * alpha * beta * px)
        )

        # now in order to project the mobius addition onto the hyperbolic disc
        # we need to divide vectors whos l2norm : |x| (not |x|^2) are higher than max norm
        maxnorm = (1.0 - PROJ_EPS) / tf.sqrt(c)

        # we can do this also after the dot with a as its a scalar division
        project_normalized = tf.where(
            tf.greater(tf.sqrt(mobaddnorm), maxnorm),  # condition
            x=maxnorm / tf.maximum(tf.sqrt(mobaddnorm), EPS),  # if true
            y=tf.ones_like(mobaddnorm),
        )  # if false
        tf.debugging.check_numerics(project_normalized, 'project_normalized nan')
        # tf.summary.histogram('project_normalized', project_normalized)

        mobaddnormprojected = tf.where(
            tf.less(tf.sqrt(mobaddnorm), maxnorm),  # condition
            x=mobaddnorm,  # if true
            y=tf.ones_like(mobaddnorm) * maxnorm ** 2,
        )
        tf.debugging.check_numerics(mobaddnormprojected, 'mobaddnormprojected nan')
        # tf.summary.histogram('mobaddnormprojected', mobaddnormprojected)

        xdota = beta * cross_correlate(inputs, filters=A_kernel)  # sh [B,H,W,ch]
        pdota = (
                alpha * tf.reduce_sum(-P_mlr * normed_A, axis=1)[None, None, None, :]
        )  # ncls
        mobdota = xdota + pdota  # sh B,H,W,ch
        mobdota *= project_normalized  # equiv to project mob add to max norm before dot

        lamb_px = 2.0 / tf.maximum(1 - c * mobaddnormprojected, EPS)
        lamb_px = tf.debugging.check_numerics(lamb_px, 'lamb_px nan')

        sineterm = tf.sqrt(c) * mobdota * lamb_px
        return 2.0 / tf.sqrt(c) * A_norm * tf.asinh(sineterm)
