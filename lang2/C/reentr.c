/* -*- buffer-read-only: t -*-
 *
 *    reentr.c
 *
 *    Copyright (C) 2002, 2003, 2005, 2006, 2007 by Larry Wall and others
 *
 *    You may distribute under the terms of either the GNU General Public
 *    License or the Artistic License, as specified in the README file.
 *
 * !!!!!!!   DO NOT EDIT THIS FILE   !!!!!!!
 * This file is built by regen/reentr.pl from data in regen/reentr.pl.
 * Any changes made here will be lost!
 */

/*
 * "Saruman," I said, standing away from him, "only one hand at a time can
 *  wield the One, and you know that well, so do not trouble to say we!"
 *
 *     [p.260 of _The Lord of the Rings_, II/ii: "The Council of Elrond"]
 */

/*
 * This file contains a collection of automatically created wrappers
 * (created by running reentr.pl) for reentrant (thread-safe) versions of
 * various library calls, such as getpwent_r.  The wrapping is done so
 * that other files like pp_sys.c calling those library functions need not
 * care about the differences between various platforms' idiosyncrasies
 * regarding these reentrant interfaces.
 */

#include "EXTERN.h"
#define PERL_IN_REENTR_C
#include "perl.h"
#include "reentr.h"
#include "keywords.h"

#define RenewDouble(data_pointer, size_pointer, type) \
    STMT_START { \
        const size_t size = MAX(*(size_pointer), 1) * 2; \
        Renew((data_pointer), (size), type); \
        *(size_pointer) = size; \
    } STMT_END

void
Perl_reentrant_size(pTHX) {
        PERL_UNUSED_CONTEXT;

        /* Set the sizes of the reentrant buffers */

#ifdef USE_REENTRANT_API
#  define REENTRANTSMALLSIZE	 256	/* Make something up. */
#  define REENTRANTUSUALSIZE	4096	/* Make something up. */

#  ifdef HAS_ASCTIME_R
        PL_reentrant_buffer->_asctime_size = 26;
#  endif /* HAS_ASCTIME_R */

#  ifdef HAS_CRYPT_R
#  endif /* HAS_CRYPT_R */

#  ifdef HAS_CTIME_R
        PL_reentrant_buffer->_ctime_size = 26;
#  endif /* HAS_CTIME_R */

#  ifdef HAS_GETGRNAM_R
#    if defined(HAS_SYSCONF) && defined(_SC_GETGR_R_SIZE_MAX) && !defined(__GLIBC__)
        PL_reentrant_buffer->_grent_size = sysconf(_SC_GETGR_R_SIZE_MAX);
        if (PL_reentrant_buffer->_grent_size == (size_t) -1)
                PL_reentrant_buffer->_grent_size = REENTRANTUSUALSIZE;
#    elif defined(__osf__) && defined(__alpha) && defined(SIABUFSIZ)
        PL_reentrant_buffer->_grent_size = SIABUFSIZ;
#    elif defined(__sgi)
        PL_reentrant_buffer->_grent_size = BUFSIZ;
#    else
        PL_reentrant_buffer->_grent_size = REENTRANTUSUALSIZE;
#    endif
#  endif /* HAS_GETGRNAM_R */

#  ifdef HAS_GETHOSTBYNAME_R
#  if !(GETHOSTBYNAME_R_PROTO == REENTRANT_PROTO_I_CSD)
        PL_reentrant_buffer->_hostent_size = REENTRANTUSUALSIZE;
#  endif
#  endif /* HAS_GETHOSTBYNAME_R */

#  ifdef HAS_GETLOGIN_R
        PL_reentrant_buffer->_getlogin_size = REENTRANTSMALLSIZE;
#  endif /* HAS_GETLOGIN_R */

#  ifdef HAS_GETNETBYNAME_R
#  if !(GETNETBYNAME_R_PROTO == REENTRANT_PROTO_I_CSD)
        PL_reentrant_buffer->_netent_size = REENTRANTUSUALSIZE;
#  endif
#  endif /* HAS_GETNETBYNAME_R */

#  ifdef HAS_GETPROTOBYNAME_R
#  if !(GETPROTOBYNAME_R_PROTO == REENTRANT_PROTO_I_CSD)
        PL_reentrant_buffer->_protoent_size = REENTRANTUSUALSIZE;
#  endif
#  endif /* HAS_GETPROTOBYNAME_R */

#  ifdef HAS_GETPWNAM_R
#    if defined(HAS_SYSCONF) && defined(_SC_GETPW_R_SIZE_MAX) && !defined(__GLIBC__)
        PL_reentrant_buffer->_pwent_size = sysconf(_SC_GETPW_R_SIZE_MAX);
        if (PL_reentrant_buffer->_pwent_size == (size_t) -1)
                PL_reentrant_buffer->_pwent_size = REENTRANTUSUALSIZE;
#    elif defined(__osf__) && defined(__alpha) && defined(SIABUFSIZ)
        PL_reentrant_buffer->_pwent_size = SIABUFSIZ;
#    elif defined(__sgi)
        PL_reentrant_buffer->_pwent_size = BUFSIZ;
#    else
        PL_reentrant_buffer->_pwent_size = REENTRANTUSUALSIZE;
#    endif
#  endif /* HAS_GETPWNAM_R */

#  ifdef HAS_GETSERVBYNAME_R
#  if !(GETSERVBYNAME_R_PROTO == REENTRANT_PROTO_I_CCSD)
        PL_reentrant_buffer->_servent_size = REENTRANTUSUALSIZE;
#  endif
#  endif /* HAS_GETSERVBYNAME_R */

#  ifdef HAS_GETSPNAM_R
#    if defined(HAS_SYSCONF) && defined(_SC_GETPW_R_SIZE_MAX) && !defined(__GLIBC__)
        PL_reentrant_buffer->_spent_size = sysconf(_SC_GETPW_R_SIZE_MAX);
        if (PL_reentrant_buffer->_spent_size == (size_t) -1)
                PL_reentrant_buffer->_spent_size = REENTRANTUSUALSIZE;
#    elif defined(__osf__) && defined(__alpha) && defined(SIABUFSIZ)
        PL_reentrant_buffer->_spent_size = SIABUFSIZ;
#    elif defined(__sgi)
        PL_reentrant_buffer->_spent_size = BUFSIZ;
#    else
        PL_reentrant_buffer->_spent_size = REENTRANTUSUALSIZE;
#    endif
#  endif /* HAS_GETSPNAM_R */

#  ifdef HAS_GMTIME_R
#  endif /* HAS_GMTIME_R */

#  ifdef HAS_LOCALTIME_R
#  endif /* HAS_LOCALTIME_R */

#  ifdef HAS_READDIR_R
        /* This is the size Solaris recommends.
         * (though we go static, should use pathconf() instead) */
        PL_reentrant_buffer->_readdir_size = sizeof(struct dirent) + MAXPATHLEN + 1;
#  endif /* HAS_READDIR_R */

#  ifdef HAS_READDIR64_R
        /* This is the size Solaris recommends.
         * (though we go static, should use pathconf() instead) */
        PL_reentrant_buffer->_readdir64_size = sizeof(struct dirent64) + MAXPATHLEN + 1;
#  endif /* HAS_READDIR64_R */

#  ifdef HAS_SETLOCALE_R
        PL_reentrant_buffer->_setlocale_size = REENTRANTSMALLSIZE;
#  endif /* HAS_SETLOCALE_R */

#  ifdef HAS_STRERROR_R
        PL_reentrant_buffer->_strerror_size = REENTRANTSMALLSIZE;
#  endif /* HAS_STRERROR_R */

#  ifdef HAS_TTYNAME_R
        PL_reentrant_buffer->_ttyname_size = REENTRANTSMALLSIZE;
#  endif /* HAS_TTYNAME_R */


#endif /* USE_REENTRANT_API */

}

void
Perl_reentrant_init(pTHX) {
        PERL_UNUSED_CONTEXT;

        /* Initialize the whole thing */

#ifdef USE_REENTRANT_API

        Newx(PL_reentrant_buffer, 1, REENTR);
        Perl_reentrant_size(aTHX);

#  ifdef HAS_ASCTIME_R
        Newx(PL_reentrant_buffer->_asctime_buffer, PL_reentrant_buffer->_asctime_size, char);
#  endif /* HAS_ASCTIME_R */

#  ifdef HAS_CRYPT_R
#  if CRYPT_R_PROTO != REENTRANT_PROTO_B_CCD
        PL_reentrant_buffer->_crypt_struct_buffer = 0;
#  endif
#  endif /* HAS_CRYPT_R */

#  ifdef HAS_CTIME_R
        Newx(PL_reentrant_buffer->_ctime_buffer, PL_reentrant_buffer->_ctime_size, char);
#  endif /* HAS_CTIME_R */

#  ifdef HAS_GETGRNAM_R
#   ifdef USE_GRENT_FPTR
        PL_reentrant_buffer->_grent_fptr = NULL;
#   endif
        Newx(PL_reentrant_buffer->_grent_buffer, PL_reentrant_buffer->_grent_size, char);
#  endif /* HAS_GETGRNAM_R */

#  ifdef HAS_GETHOSTBYNAME_R
#  if !(GETHOSTBYNAME_R_PROTO == REENTRANT_PROTO_I_CSD)
        Newx(PL_reentrant_buffer->_hostent_buffer, PL_reentrant_buffer->_hostent_size, char);
#  endif
#  endif /* HAS_GETHOSTBYNAME_R */

#  ifdef HAS_GETLOGIN_R
        Newx(PL_reentrant_buffer->_getlogin_buffer, PL_reentrant_buffer->_getlogin_size, char);
#  endif /* HAS_GETLOGIN_R */

#  ifdef HAS_GETNETBYNAME_R
#  if !(GETNETBYNAME_R_PROTO == REENTRANT_PROTO_I_CSD)
        Newx(PL_reentrant_buffer->_netent_buffer, PL_reentrant_buffer->_netent_size, char);
#  endif
#  endif /* HAS_GETNETBYNAME_R */

#  ifdef HAS_GETPROTOBYNAME_R
#  if !(GETPROTOBYNAME_R_PROTO == REENTRANT_PROTO_I_CSD)
        Newx(PL_reentrant_buffer->_protoent_buffer, PL_reentrant_buffer->_protoent_size, char);
#  endif
#  endif /* HAS_GETPROTOBYNAME_R */

#  ifdef HAS_GETPWNAM_R
#   ifdef USE_PWENT_FPTR
        PL_reentrant_buffer->_pwent_fptr = NULL;
#   endif
        Newx(PL_reentrant_buffer->_pwent_buffer, PL_reentrant_buffer->_pwent_size, char);
#  endif /* HAS_GETPWNAM_R */

#  ifdef HAS_GETSERVBYNAME_R
#  if !(GETSERVBYNAME_R_PROTO == REENTRANT_PROTO_I_CCSD)
        Newx(PL_reentrant_buffer->_servent_buffer, PL_reentrant_buffer->_servent_size, char);
#  endif
#  endif /* HAS_GETSERVBYNAME_R */

#  ifdef HAS_GETSPNAM_R
#   ifdef USE_SPENT_FPTR
        PL_reentrant_buffer->_spent_fptr = NULL;
#   endif
        Newx(PL_reentrant_buffer->_spent_buffer, PL_reentrant_buffer->_spent_size, char);
#  endif /* HAS_GETSPNAM_R */

#  ifdef HAS_GMTIME_R
#  endif /* HAS_GMTIME_R */

#  ifdef HAS_LOCALTIME_R
#  endif /* HAS_LOCALTIME_R */

#  ifdef HAS_READDIR_R
        PL_reentrant_buffer->_readdir_struct = (struct dirent*)safemalloc(PL_reentrant_buffer->_readdir_size);
#  endif /* HAS_READDIR_R */

#  ifdef HAS_READDIR64_R
        PL_reentrant_buffer->_readdir64_struct = (struct dirent64*)safemalloc(PL_reentrant_buffer->_readdir64_size);
#  endif /* HAS_READDIR64_R */

#  ifdef HAS_SETLOCALE_R
        Newx(PL_reentrant_buffer->_setlocale_buffer, PL_reentrant_buffer->_setlocale_size, char);
#  endif /* HAS_SETLOCALE_R */

#  ifdef HAS_STRERROR_R
        Newx(PL_reentrant_buffer->_strerror_buffer, PL_reentrant_buffer->_strerror_size, char);
#  endif /* HAS_STRERROR_R */

#  ifdef HAS_TTYNAME_R
        Newx(PL_reentrant_buffer->_ttyname_buffer, PL_reentrant_buffer->_ttyname_size, char);
#  endif /* HAS_TTYNAME_R */


#endif /* USE_REENTRANT_API */

}

void
Perl_reentrant_free(pTHX) {
        PERL_UNUSED_CONTEXT;

        /* Tear down */

#ifdef USE_REENTRANT_API

#  ifdef HAS_ASCTIME_R
        Safefree(PL_reentrant_buffer->_asctime_buffer);
#  endif /* HAS_ASCTIME_R */

#  ifdef HAS_CRYPT_R
#  if CRYPT_R_PROTO != REENTRANT_PROTO_B_CCD
        Safefree(PL_reentrant_buffer->_crypt_struct_buffer);
#  endif
#  endif /* HAS_CRYPT_R */

#  ifdef HAS_CTIME_R
        Safefree(PL_reentrant_buffer->_ctime_buffer);
#  endif /* HAS_CTIME_R */

#  ifdef HAS_GETGRNAM_R
        Safefree(PL_reentrant_buffer->_grent_buffer);
#  endif /* HAS_GETGRNAM_R */

#  ifdef HAS_GETHOSTBYNAME_R
#  if !(GETHOSTBYNAME_R_PROTO == REENTRANT_PROTO_I_CSD)
        Safefree(PL_reentrant_buffer->_hostent_buffer);
#  endif
#  endif /* HAS_GETHOSTBYNAME_R */

#  ifdef HAS_GETLOGIN_R
        Safefree(PL_reentrant_buffer->_getlogin_buffer);
#  endif /* HAS_GETLOGIN_R */

#  ifdef HAS_GETNETBYNAME_R
#  if !(GETNETBYNAME_R_PROTO == REENTRANT_PROTO_I_CSD)
        Safefree(PL_reentrant_buffer->_netent_buffer);
#  endif
#  endif /* HAS_GETNETBYNAME_R */

#  ifdef HAS_GETPROTOBYNAME_R
#  if !(GETPROTOBYNAME_R_PROTO == REENTRANT_PROTO_I_CSD)
        Safefree(PL_reentrant_buffer->_protoent_buffer);
#  endif
#  endif /* HAS_GETPROTOBYNAME_R */

#  ifdef HAS_GETPWNAM_R
        Safefree(PL_reentrant_buffer->_pwent_buffer);
#  endif /* HAS_GETPWNAM_R */

#  ifdef HAS_GETSERVBYNAME_R
#  if !(GETSERVBYNAME_R_PROTO == REENTRANT_PROTO_I_CCSD)
        Safefree(PL_reentrant_buffer->_servent_buffer);
#  endif
#  endif /* HAS_GETSERVBYNAME_R */

#  ifdef HAS_GETSPNAM_R
        Safefree(PL_reentrant_buffer->_spent_buffer);
#  endif /* HAS_GETSPNAM_R */

#  ifdef HAS_GMTIME_R
#  endif /* HAS_GMTIME_R */

#  ifdef HAS_LOCALTIME_R
#  endif /* HAS_LOCALTIME_R */

#  ifdef HAS_READDIR_R
        Safefree(PL_reentrant_buffer->_readdir_struct);
#  endif /* HAS_READDIR_R */

#  ifdef HAS_READDIR64_R
        Safefree(PL_reentrant_buffer->_readdir64_struct);
#  endif /* HAS_READDIR64_R */

#  ifdef HAS_SETLOCALE_R
        Safefree(PL_reentrant_buffer->_setlocale_buffer);
#  endif /* HAS_SETLOCALE_R */

#  ifdef HAS_STRERROR_R
        Safefree(PL_reentrant_buffer->_strerror_buffer);
#  endif /* HAS_STRERROR_R */

#  ifdef HAS_TTYNAME_R
        Safefree(PL_reentrant_buffer->_ttyname_buffer);
#  endif /* HAS_TTYNAME_R */


        Safefree(PL_reentrant_buffer);

#endif /* USE_REENTRANT_API */
}

void*
Perl_reentrant_retry(const char *f, ...)
{
    /* This function is set up to be called if the normal function returns
     * failure with errno ERANGE, which indicates the buffer is too small.
     * This function calls the failing one again with a larger buffer.
     *
     * What has happened is that, due to the magic of C preprocessor macro
     * expansion, when the original code called function 'foo(args)', it was
     * instead compiled into something like a call of 'foo_r(args, buffer)'
     * Below we retry with 'foo', but the preprocessor has changed that into
     * 'foo_r', so this function will end up calling itself recursively, each
     * time with a larger buffer.  If PERL_REENTRANT_MAXSIZE is defined, it
     * won't increase beyond that, instead failing. */

    void *retptr = NULL;
    va_list ap;

    I32 key = 0;

#ifdef USE_REENTRANT_API

    dTHX;

    key = Perl_keyword (aTHX_ f, strlen(f), FALSE /* not feature enabled */);

    /* Easier to special case this here than in embed.pl. (Look at what it
       generates for proto.h) */
    PERL_ARGS_ASSERT_REENTRANT_RETRY;

#endif

    if (key == 0) {

#ifdef HAS_GETSPNAM_R

        /* This is a #define as has no corresponding keyword */
        if (strEQ(f, "getspnam")) {
            key = KEY_getspnam;
        }

#endif

    }
    else if (key < 0) {
        key = -key;
    }

    va_start(ap, f);

#ifdef USE_REENTRANT_API

    switch (key) {

#  ifdef USE_HOSTENT_BUFFER

    case KEY_gethostbyaddr:
    case KEY_gethostbyname:
    case KEY_endhostent:
        {
            char * host_addr;
            Size_t asize;
            char * host_name;
            int anint;

#    ifdef PERL_REENTRANT_MAXSIZE
            if (PL_reentrant_buffer->_hostent_size <=
                PERL_REENTRANT_MAXSIZE / 2)
#    endif
            RenewDouble(PL_reentrant_buffer->_hostent_buffer,
                    &PL_reentrant_buffer->_hostent_size, char);
            switch (key) {
                case KEY_gethostbyaddr:
                    host_addr = va_arg(ap, char *);
                    asize = va_arg(ap, Size_t);
                    anint  = va_arg(ap, int);
                    /* socklen_t is what Posix 2001 says this should be */
                    retptr = gethostbyaddr(host_addr, (socklen_t) asize, anint); break;
                case KEY_gethostbyname:
                    host_name = va_arg(ap, char *);
                    retptr = gethostbyname(host_name); break;
                case KEY_endhostent:
                    retptr = gethostent(); break;
                default:
                    SETERRNO(ERANGE, LIB_INVARG);
                    break;
            }
        }
        break;

#  endif
#  ifdef USE_GRENT_BUFFER

    case KEY_getgrent:
    case KEY_getgrgid:
    case KEY_getgrnam:
        {
            char * name;
            Gid_t gid;

#    ifdef PERL_REENTRANT_MAXSIZE
            if (PL_reentrant_buffer->_grent_size <=
                PERL_REENTRANT_MAXSIZE / 2)
#    endif
            RenewDouble(PL_reentrant_buffer->_grent_buffer,
                    &PL_reentrant_buffer->_grent_size, char);
            switch (key) {
                case KEY_getgrnam:
                    name = va_arg(ap, char *);
                    retptr = getgrnam(name); break;
                case KEY_getgrgid:
#    if Gid_t_size < INTSIZE
                    gid = (Gid_t)va_arg(ap, int);
#    else
                    gid = va_arg(ap, Gid_t);
#    endif
                    retptr = getgrgid(gid); break;
                case KEY_getgrent:
                    retptr = getgrent(); break;
                default:
                    SETERRNO(ERANGE, LIB_INVARG);
                    break;
            }
        }
        break;

#  endif
#  ifdef USE_NETENT_BUFFER

    case KEY_getnetbyaddr:
    case KEY_getnetbyname:
    case KEY_getnetent:
        {
            char * name;
            Netdb_net_t net;
            int anint;

#    ifdef PERL_REENTRANT_MAXSIZE
            if (PL_reentrant_buffer->_netent_size <=
                PERL_REENTRANT_MAXSIZE / 2)
#    endif
            RenewDouble(PL_reentrant_buffer->_netent_buffer,
                    &PL_reentrant_buffer->_netent_size, char);
            switch (key) {
                case KEY_getnetbyaddr:
                    net = va_arg(ap, Netdb_net_t);
                    anint = va_arg(ap, int);
                    retptr = getnetbyaddr(net, anint); break;
                case KEY_getnetbyname:
                    name = va_arg(ap, char *);
                    retptr = getnetbyname(name); break;
                case KEY_getnetent:
                    retptr = getnetent(); break;
                default:
                    SETERRNO(ERANGE, LIB_INVARG);
                    break;
            }
        }
        break;

#  endif
#  ifdef USE_PWENT_BUFFER

    case  KEY_getpwnam:
    case  KEY_getpwuid:
    case  KEY_getpwent:
        {
            Uid_t uid;
            char * name;

#    ifdef PERL_REENTRANT_MAXSIZE
            if (PL_reentrant_buffer->_pwent_size <=
                PERL_REENTRANT_MAXSIZE / 2)

#    endif
            RenewDouble(PL_reentrant_buffer->_pwent_buffer,
                    &PL_reentrant_buffer->_pwent_size, char);
            switch (key) {
                case KEY_getpwnam:
                    name = va_arg(ap, char *);
                    retptr = getpwnam(name); break;
                case KEY_getpwuid:

#    if Uid_t_size < INTSIZE
                    uid = (Uid_t)va_arg(ap, int);
#    else
                    uid = va_arg(ap, Uid_t);
#    endif
                    retptr = getpwuid(uid); break;

#  if defined(HAS_GETPWENT) || defined(HAS_GETPWENT_R)

                case KEY_getpwent:
                    retptr = getpwent(); break;
#  endif
                default:
                    SETERRNO(ERANGE, LIB_INVARG);
                    break;
            }
        }
        break;

#  endif
#  ifdef USE_SPENT_BUFFER

    case KEY_getspnam:
        {
            char * name;

#    ifdef PERL_REENTRANT_MAXSIZE
            if (PL_reentrant_buffer->_spent_size <=
                PERL_REENTRANT_MAXSIZE / 2)

#    endif
            RenewDouble(PL_reentrant_buffer->_spent_buffer,
                    &PL_reentrant_buffer->_spent_size, char);
            switch (key) {
                case KEY_getspnam:
                    name = va_arg(ap, char *);
                    retptr = getspnam(name); break;
                default:
                    SETERRNO(ERANGE, LIB_INVARG);
                    break;
            }
        }
        break;

#  endif
#  ifdef USE_PROTOENT_BUFFER

    case KEY_getprotobyname:
    case KEY_getprotobynumber:
    case KEY_getprotoent:
        {
            char * name;
            int anint;

#    ifdef PERL_REENTRANT_MAXSIZE
            if (PL_reentrant_buffer->_protoent_size <=
                PERL_REENTRANT_MAXSIZE / 2)
#    endif
            RenewDouble(PL_reentrant_buffer->_protoent_buffer,
                    &PL_reentrant_buffer->_protoent_size, char);
            switch (key) {
                case KEY_getprotobyname:
                    name = va_arg(ap, char *);
                    retptr = getprotobyname(name); break;
                case KEY_getprotobynumber:
                    anint = va_arg(ap, int);
                    retptr = getprotobynumber(anint); break;
                case KEY_getprotoent:
                    retptr = getprotoent(); break;
                default:
                    SETERRNO(ERANGE, LIB_INVARG);
                    break;
            }
        }
        break;

#  endif
#  ifdef USE_SERVENT_BUFFER

    case KEY_getservbyname:
    case KEY_getservbyport:
    case KEY_getservent:
        {
            char * name;
            char * proto;
            int anint;

#    ifdef PERL_REENTRANT_MAXSIZE
            if (PL_reentrant_buffer->_servent_size <=
                PERL_REENTRANT_MAXSIZE / 2)
#    endif
            RenewDouble(PL_reentrant_buffer->_servent_buffer,
                    &PL_reentrant_buffer->_servent_size, char);
            switch (key) {
                case KEY_getservbyname:
                    name = va_arg(ap, char *);
                    proto = va_arg(ap, char *);
                    retptr = getservbyname(name, proto); break;
                case KEY_getservbyport:
                    anint = va_arg(ap, int);
                    name = va_arg(ap, char *);
                    retptr = getservbyport(anint, name); break;
                case KEY_getservent:
                    retptr = getservent(); break;
                default:
                    SETERRNO(ERANGE, LIB_INVARG);
                    break;
            }
        }
        break;

#  endif

    default:
        /* Not known how to retry, so just fail. */
        break;
    }

#else

    PERL_UNUSED_ARG(f);

#endif

    va_end(ap);
    return retptr;
}

/* ex: set ro: */
