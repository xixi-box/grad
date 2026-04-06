import { motion } from 'framer-motion'
import { useTranslation } from 'react-i18next'
import { Github, Book, MessageCircle, MessageSquare, Heart } from 'lucide-react'

export default function Footer() {
  const { t } = useTranslation()

  const links = [
    { name: t('footer.links.github'), icon: Github, href: 'https://github.com/xixi-box/grad' },
    { name: t('footer.links.docs'), icon: Book, href: 'https://github.com/xixi-box/grad#readme' },
    { name: t('footer.links.issues'), icon: MessageCircle, href: 'https://github.com/xixi-box/grad/issues' },
    { name: t('footer.links.discussions'), icon: MessageSquare, href: 'https://github.com/xixi-box/grad/discussions' },
  ]

  return (
    <footer className="py-12 border-t border-slate-800">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex flex-col md:flex-row items-center justify-between gap-6">
          {/* Logo */}
          <motion.div
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            className="flex items-center gap-2"
          >
            <span className="gradient-text font-bold">DUSt3R</span>
            <span className="text-slate-500">→</span>
            <span className="gradient-text font-bold">3DGS</span>
          </motion.div>

          {/* Links */}
          <div className="flex items-center gap-6">
            {links.map((link) => (
              <motion.a
                key={link.name}
                href={link.href}
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-2 text-slate-400 hover:text-slate-50 transition-colors"
                whileHover={{ y: -2 }}
              >
                <link.icon size={18} />
                <span className="text-sm hidden sm:inline">{link.name}</span>
              </motion.a>
            ))}
          </div>
        </div>

        {/* Bottom */}
        <div className="mt-8 pt-8 border-t border-slate-800 flex flex-col sm:flex-row items-center justify-between gap-4 text-sm text-slate-500">
          <p className="flex items-center gap-1">
            {t('footer.madeWith')} <Heart size={14} className="text-red-500" /> {t('footer.by')}{' '}
            <a
              href="https://github.com/xixi-box"
              target="_blank"
              rel="noopener noreferrer"
              className="text-violet-400 hover:text-violet-300"
            >
              xixi-box
            </a>
          </p>
          <p>{t('footer.license')}</p>
        </div>
      </div>
    </footer>
  )
}